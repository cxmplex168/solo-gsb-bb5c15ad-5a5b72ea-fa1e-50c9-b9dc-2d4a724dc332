"""Cancel-impact preview (GET /jobs/{id}/cancel-impact) and cascade
cancellation (POST /jobs/{id}/cancel-cascade).

Covers: the transitive closure over the cross-tenant dependency graph
(through CANCELLED nodes, never through COMPLETED/RUNNING ones), one
shortest dependency chain per affected job with id-sorted results,
read-only preview semantics, the cascade cancelling the target plus every
still-QUEUED job in scope (releasing a RUNNING target's GPUs, leaving
already-CANCELLED intermediates untouched, resetting drained tenants'
aging clocks, recomputing the scope inside the write transaction), the
404/409 mappings, and the HTTP surface.
"""
from __future__ import annotations

import pytest

from app import db, scheduler
from tests.conftest import at, iso  # noqa: F401


def submit(tenant, gpus=1, minutes=10, weight=1.0, when=None, depends_on=None):
    job = scheduler.submit_job(
        tenant, gpus, minutes, weight, depends_on=depends_on
    )
    if when is not None:
        db.get_conn().execute(
            "UPDATE jobs SET queued_at = ? WHERE id = ?", (when, job["id"])
        )
    return scheduler.get_job(job["id"])


def statuses():
    return {j["id"]: j["status"] for j in scheduler.list_jobs()}


def affected_ids(report):
    return [j["id"] for j in report["affected_jobs"]]


def chains(report):
    return {j["id"]: j["chain"] for j in report["affected_jobs"]}


# ---------------------------------------------------------------------------
# Impact preview (service level)
# ---------------------------------------------------------------------------

def test_impact_linear_chain_cross_tenant(conn):
    j1 = submit("team-a", when=iso(0))                       # J000001
    j2 = submit("team-b", when=iso(1), depends_on=[j1["id"]])  # J000002
    j3 = submit("team-c", when=iso(2), depends_on=[j2["id"]])  # J000003

    report = scheduler.cancel_impact(j1["id"])
    assert report["job_id"] == j1["id"]
    assert report["status"] == "QUEUED"
    assert report["affected_count"] == 2
    # Sorted by job id; each entry carries one shortest chain from the target.
    assert affected_ids(report) == [j2["id"], j3["id"]]
    assert chains(report) == {
        j2["id"]: [j1["id"], j2["id"]],
        j3["id"]: [j1["id"], j2["id"], j3["id"]],
    }
    # Affected entries are full job objects (plus the chain).
    entry = report["affected_jobs"][0]
    assert entry["tenant_id"] == "team-b"
    assert entry["status"] == "QUEUED"
    assert entry["depends_on"] == [j1["id"]]


def test_impact_passes_through_cancelled_nodes(conn):
    j1 = submit("a")                            # J000001
    j2 = submit("a", depends_on=[j1["id"]])     # J000002
    j3 = submit("b", depends_on=[j2["id"]])     # J000003
    scheduler.cancel_job(j2["id"])              # intermediate node cancelled
    cancelled_at = scheduler.get_job(j2["id"])["completed_at"]

    report = scheduler.cancel_impact(j1["id"])
    # The CANCELLED intermediate is traversed but not itself reported.
    assert affected_ids(report) == [j3["id"]]
    assert chains(report)[j3["id"]] == [j1["id"], j2["id"], j3["id"]]
    # The preview did not touch the already-cancelled node.
    assert scheduler.get_job(j2["id"])["completed_at"] == cancelled_at


def test_impact_diamond_reports_one_shortest_chain(conn):
    j1 = submit("a")                            # J000001
    j2 = submit("a", depends_on=[j1["id"]])     # J000002
    j3 = submit("b", depends_on=[j1["id"]])     # J000003
    j4 = submit("c", depends_on=[j2["id"], j3["id"]])  # J000004

    report = scheduler.cancel_impact(j1["id"])
    assert affected_ids(report) == [j2["id"], j3["id"], j4["id"]]
    # Two equally short chains reach J000004; BFS with id-ordered neighbours
    # deterministically reports the one through J000002.
    assert chains(report)[j4["id"]] == [j1["id"], j2["id"], j4["id"]]


def test_impact_excludes_unrelated_and_released_jobs(conn):
    done = submit("base", when=iso(0))          # J000001
    scheduler.run_batch(now=at(1))              # J000001 -> RUNNING
    scheduler.complete_job(done["id"])          # J000001 -> COMPLETED
    released = submit("base", depends_on=[done["id"]])  # J000002, schedulable
    target = submit("t", when=iso(2))           # J000003
    doomed = submit("t", depends_on=[target["id"]])     # J000004
    independent = submit("other")               # J000005

    report = scheduler.cancel_impact(target["id"])
    assert affected_ids(report) == [doomed["id"]]
    assert released["id"] not in affected_ids(report)
    assert independent["id"] not in affected_ids(report)


def test_impact_running_target_allowed(conn):
    j1 = submit("a", gpus=2, when=iso(0))       # J000001
    j2 = submit("b", depends_on=[j1["id"]])     # J000002
    scheduler.run_batch(now=at(1))              # J000001 -> RUNNING
    report = scheduler.cancel_impact(j1["id"])
    assert report["status"] == "RUNNING"
    assert affected_ids(report) == [j2["id"]]


def test_impact_no_dependents(conn):
    j1 = submit("a")
    report = scheduler.cancel_impact(j1["id"])
    assert report["affected_count"] == 0
    assert report["affected_jobs"] == []


def test_impact_unknown_target_404(conn):
    with pytest.raises(scheduler.NotFound) as exc:
        scheduler.cancel_impact("J000999")
    assert exc.value.status_code == 404


def test_impact_completed_or_cancelled_target_409(conn):
    done = submit("a", when=iso(0))
    scheduler.run_batch(now=at(1))
    scheduler.complete_job(done["id"])
    with pytest.raises(scheduler.Conflict) as exc:
        scheduler.cancel_impact(done["id"])
    assert exc.value.status_code == 409

    cancelled = submit("a")
    scheduler.cancel_job(cancelled["id"])
    with pytest.raises(scheduler.Conflict):
        scheduler.cancel_impact(cancelled["id"])


def test_impact_preview_writes_nothing(conn):
    j1 = submit("a", when=iso(0))               # J000001
    j2 = submit("b", depends_on=[j1["id"]])     # J000002
    j3 = submit("c", depends_on=[j2["id"]])     # J000003
    scheduler.run_batch(now=at(1))              # settle some state

    def snapshot():
        conn = db.get_conn()
        jobs = conn.execute(
            "SELECT id, status, completed_at FROM jobs ORDER BY id"
        ).fetchall()
        tenants = conn.execute(
            "SELECT id, aging_settled_at, deficit FROM tenants ORDER BY id"
        ).fetchall()
        ledger = conn.execute("SELECT COUNT(*) AS n FROM ledger").fetchone()
        batches = conn.execute("SELECT COUNT(*) AS n FROM batches").fetchone()
        return (
            [(r["id"], r["status"], r["completed_at"]) for r in jobs],
            [(r["id"], r["aging_settled_at"], r["deficit"]) for r in tenants],
            ledger["n"],
            batches["n"],
        )

    before = snapshot()
    scheduler.cancel_impact(j1["id"])
    assert snapshot() == before


# ---------------------------------------------------------------------------
# Cascade cancellation (service level)
# ---------------------------------------------------------------------------

def test_cascade_cancels_target_and_queued_descendants(conn):
    j1 = submit("team-a", when=iso(0))                       # J000001
    j2 = submit("team-b", when=iso(1), depends_on=[j1["id"]])  # J000002
    j3 = submit("team-c", when=iso(2), depends_on=[j2["id"]])  # J000003
    j4 = submit("team-a", when=iso(3))                       # J000004 (other)

    result = scheduler.cancel_cascade(j1["id"])
    assert result["job_id"] == j1["id"]
    assert result["status"] == "CANCELLED"
    assert result["released_gpus"] == 0  # QUEUED target holds no GPUs
    assert result["cancelled_count"] == 3
    assert result["cancelled_job_ids"] == [j1["id"], j2["id"], j3["id"]]
    assert result["affected_count"] == 2
    assert affected_ids(result) == [j2["id"], j3["id"]]
    assert chains(result)[j3["id"]] == [j1["id"], j2["id"], j3["id"]]
    # The affected entries reflect the post-cascade state.
    assert all(j["status"] == "CANCELLED" for j in result["affected_jobs"])

    assert statuses()[j1["id"]] == "CANCELLED"
    assert statuses()[j2["id"]] == "CANCELLED"
    assert statuses()[j3["id"]] == "CANCELLED"
    assert statuses()[j4["id"]] == "QUEUED"  # independent job untouched


def test_cascade_running_target_releases_gpus(conn):
    j1 = submit("a", gpus=3, when=iso(0))       # J000001
    j2 = submit("b", depends_on=[j1["id"]])     # J000002
    scheduler.run_batch(now=at(1))              # J000001 -> RUNNING (3 GPUs)
    assert scheduler.scheduler_state()["used_gpus"] == 3

    result = scheduler.cancel_cascade(j1["id"])
    assert result["released_gpus"] == 3
    assert scheduler.scheduler_state()["used_gpus"] == 0
    assert statuses()[j2["id"]] == "CANCELLED"


def test_cascade_leaves_cancelled_intermediate_untouched(conn):
    j1 = submit("a")                            # J000001
    j2 = submit("a", depends_on=[j1["id"]])     # J000002
    j3 = submit("b", depends_on=[j2["id"]])     # J000003
    scheduler.cancel_job(j2["id"])
    cancelled_at = scheduler.get_job(j2["id"])["completed_at"]

    result = scheduler.cancel_cascade(j1["id"])
    # The already-CANCELLED intermediate is traversal-only: not re-modified
    # and not counted among the freshly cancelled jobs.
    assert result["cancelled_job_ids"] == [j1["id"], j3["id"]]
    assert result["cancelled_count"] == 2
    after = scheduler.get_job(j2["id"])
    assert after["status"] == "CANCELLED"
    assert after["completed_at"] == cancelled_at


def test_cascade_resets_aging_clock_only_for_drained_tenants(conn):
    j1 = submit("a", when=iso(0))               # J000001
    j2 = submit("b", when=iso(1), depends_on=[j1["id"]])  # J000002
    j3 = submit("b", when=iso(2))               # J000003 (independent)
    scheduler.run_batch(now=at(5))              # J000001 RUNNING; b ages
    aging_b = scheduler.get_tenant("b")["aging_settled_at"]
    assert aging_b is not None

    scheduler.cancel_cascade(j1["id"])
    # Tenant b still has J000003 queued: its aging clock is preserved.
    assert scheduler.get_tenant("b")["aging_settled_at"] == aging_b
    # Tenant a's queue drained: aging clock reset, exactly as a plain cancel.
    assert scheduler.get_tenant("a")["aging_settled_at"] is None

    # Now drain tenant b via a cascade on its remaining dependency: revise
    # nothing, just cascade the independent job's tenant-mate scenario.
    scheduler.cancel_cascade(j3["id"])
    assert scheduler.get_tenant("b")["aging_settled_at"] is None


def test_cascade_recomputes_scope_inside_transaction(conn):
    j1 = submit("a")                            # J000001
    j2 = submit("a", depends_on=[j1["id"]])     # J000002
    j3 = submit("b")                            # J000003 (initially unrelated)

    # Preview before J000003 joins the doomed subtree: only J000002.
    assert affected_ids(scheduler.cancel_impact(j1["id"])) == [j2["id"]]
    # The cascade recomputes the scope rather than reusing any earlier view.
    scheduler.revise_dependencies(j3["id"], [j2["id"]])
    result = scheduler.cancel_cascade(j1["id"])
    assert result["cancelled_job_ids"] == [j1["id"], j2["id"], j3["id"]]


def test_cascade_unknown_target_404(conn):
    with pytest.raises(scheduler.NotFound) as exc:
        scheduler.cancel_cascade("J000999")
    assert exc.value.status_code == 404


def test_cascade_completed_or_cancelled_target_409(conn):
    done = submit("a", when=iso(0))
    scheduler.run_batch(now=at(1))
    scheduler.complete_job(done["id"])
    with pytest.raises(scheduler.Conflict) as exc:
        scheduler.cancel_cascade(done["id"])
    assert exc.value.status_code == 409

    cancelled = submit("a")
    scheduler.cancel_job(cancelled["id"])
    with pytest.raises(scheduler.Conflict):
        scheduler.cancel_cascade(cancelled["id"])


def test_cascade_then_existing_views_reflect_results(conn):
    j1 = submit("team-a", gpus=2, when=iso(0))               # J000001
    j2 = submit("team-b", when=iso(1), depends_on=[j1["id"]])  # J000002
    # Too big to dispatch alongside J000001 (2 + 7 > 8), so it stays queued.
    j3 = submit("team-c", gpus=7, when=iso(2))               # J000003
    scheduler.run_batch(now=at(1))              # J000001 RUNNING
    scheduler.cancel_cascade(j1["id"])

    # Job queries reflect the cancellations.
    assert scheduler.get_job(j1["id"])["status"] == "CANCELLED"
    assert scheduler.get_job(j2["id"])["status"] == "CANCELLED"
    cancelled = scheduler.list_jobs(status="CANCELLED")
    assert {j["id"] for j in cancelled} == {j1["id"], j2["id"]}
    # The queue only shows the surviving queued job; GPUs are free again.
    queue = scheduler.queue_overview()
    assert queue["used_gpus"] == 0
    assert queue["queued_jobs"] == 1
    assert [g["tenant_id"] for g in queue["queues"]] == ["team-c"]
    assert queue["queues"][0]["jobs"][0]["id"] == j3["id"]


# ---------------------------------------------------------------------------
# HTTP level
# ---------------------------------------------------------------------------

def test_http_impact_preview(client):
    j1 = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 1, "duration_min": 5}).json()["id"]
    j2 = client.post("/jobs", json={
        "tenant_id": "team-b", "gpus": 1, "duration_min": 5,
        "depends_on": [j1]}).json()["id"]
    j3 = client.post("/jobs", json={
        "tenant_id": "team-c", "gpus": 1, "duration_min": 5,
        "depends_on": [j2]}).json()["id"]

    r = client.get(f"/jobs/{j1}/cancel-impact")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == j1
    assert body["status"] == "QUEUED"
    assert body["affected_count"] == 2
    assert [j["id"] for j in body["affected_jobs"]] == [j2, j3]
    assert body["affected_jobs"][0]["chain"] == [j1, j2]
    assert body["affected_jobs"][1]["chain"] == [j1, j2, j3]

    # Read-only: every job is still QUEUED afterwards.
    assert {j["id"] for j in client.get("/jobs?status=QUEUED").json()} == {
        j1, j2, j3}


def test_http_impact_unknown_404(client):
    r = client.get("/jobs/J000999/cancel-impact")
    assert r.status_code == 404
    assert "not found" in r.json()["error"]


def test_http_impact_finished_target_409(client):
    j = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5}).json()["id"]
    client.post(f"/jobs/{j}/cancel")
    r = client.get(f"/jobs/{j}/cancel-impact")
    assert r.status_code == 409
    assert "already cancelled" in r.json()["error"]


def test_http_cascade(client):
    j1 = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 2, "duration_min": 5}).json()["id"]
    j2 = client.post("/jobs", json={
        "tenant_id": "team-b", "gpus": 1, "duration_min": 5,
        "depends_on": [j1]}).json()["id"]
    j3 = client.post("/jobs", json={
        "tenant_id": "team-c", "gpus": 1, "duration_min": 5,
        "depends_on": [j2]}).json()["id"]
    client.post("/scheduler/batch")             # J000001 -> RUNNING (2 GPUs)
    assert client.get("/scheduler/state").json()["used_gpus"] == 2

    r = client.post(f"/jobs/{j1}/cancel-cascade")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == j1
    assert body["status"] == "CANCELLED"
    assert body["released_gpus"] == 2
    assert body["cancelled_count"] == 3
    assert body["cancelled_job_ids"] == [j1, j2, j3]
    assert [j["id"] for j in body["affected_jobs"]] == [j2, j3]
    assert body["affected_jobs"][1]["chain"] == [j1, j2, j3]

    # Existing job and queue interfaces reflect the result.
    assert client.get(f"/jobs/{j2}").json()["status"] == "CANCELLED"
    assert client.get(f"/jobs/{j3}").json()["status"] == "CANCELLED"
    assert client.get("/scheduler/state").json()["used_gpus"] == 0
    queue = client.get("/queue").json()
    assert queue["queued_jobs"] == 0
    assert queue["queues"] == []


def test_http_cascade_unknown_404(client):
    r = client.post("/jobs/J000999/cancel-cascade")
    assert r.status_code == 404
    assert "not found" in r.json()["error"]


def test_http_cascade_finished_target_409(client):
    done = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5}).json()["id"]
    client.post("/scheduler/batch")             # -> RUNNING
    client.post(f"/jobs/{done}/complete")
    assert client.post(f"/jobs/{done}/cancel-cascade").status_code == 409

    cancelled = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5}).json()["id"]
    client.post(f"/jobs/{cancelled}/cancel")
    r = client.post(f"/jobs/{cancelled}/cancel-cascade")
    assert r.status_code == 409
    assert "already cancelled" in r.json()["error"]
