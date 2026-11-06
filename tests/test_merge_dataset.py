"""merge_dataset.plan_merge: clips from another laptop -> safe destinations here."""

import merge_dataset as md

SIGNS = {"circle": "shapes", "good_morning": "greetings"}


def row(signer, sign_id, take, ts, status="kept", category="shapes"):
    return {"timestamp": ts, "signer": signer, "signing_hand": "right", "category": category,
            "sign_id": sign_id, "take": str(take), "n_frames": "90", "camera_name": "cam",
            "file_path": f"dataset/raw/{category}/{sign_id}/{signer}/{sign_id}_{signer}_{take:02d}.mp4",
            "status": status}


def test_free_take_numbers_are_kept():
    plans, skipped, already = md.plan_merge([row("S3", "circle", 1, "t1")], [], SIGNS)
    assert [p.dest for p in plans] == ["dataset/raw/shapes/circle/S3/circle_S3_01.mp4"]
    assert skipped == [] and already == 0


def test_clashing_take_numbers_continue_after_the_highest():
    local = [row("S3", "circle", 1, "a"), row("S3", "circle", 2, "b", status="needs_rerecord")]
    source = [row("S3", "circle", 1, "x"), row("S3", "circle", 2, "y"), row("S3", "circle", 5, "z")]
    plans, _, _ = md.plan_merge(source, local, SIGNS)
    assert [p.row["take"] for p in plans] == ["3", "4", "5"]
    assert plans[0].row["file_path"] == "dataset/raw/shapes/circle/S3/circle_S3_03.mp4"


def test_files_on_disk_without_a_row_are_never_overwritten():
    plans, _, _ = md.plan_merge([row("S2", "circle", 1, "x")], [], SIGNS,
                                disk_takes=lambda c, s, g: {1, 2})
    assert plans[0].row["take"] == "3"


def test_rerun_skips_already_merged_and_pub_is_ignored():
    source = [row("S2", "circle", 1, "x"), row("PUB", "circle", 1, "p")]
    first, _, _ = md.plan_merge(source, [], SIGNS)
    assert len(first) == 1
    again, _, already = md.plan_merge(source, [p.row for p in first], SIGNS)
    assert again == [] and already == 1


def test_rename_only_and_unknown_signs():
    source = [row("S1", "circle", 1, "x"), row("S3", "circle", 1, "y"), row("S1", "pentagon", 1, "z")]
    plans, skipped, _ = md.plan_merge(source, [], SIGNS, rename={"S1": "S2"}, only={"S2"})
    assert [p.dest for p in plans] == ["dataset/raw/shapes/circle/S2/circle_S2_01.mp4"]
    assert plans[0].row["signer"] == "S2" and plans[0].src.endswith("circle_S1_01.mp4")
    assert len(skipped) == 1 and "pentagon" in skipped[0][1]


# --- only some categories --------------------------------------------------------
def test_categories_keeps_only_the_chosen_ones_and_reports_nothing_about_the_rest():
    source = [row("S2", "circle", 1, "x"), row("S2", "good_morning", 1, "y", category="greetings"),
              row("S2", "pentagon", 1, "z")]                       # unknown sign, row says shapes
    plans, skipped, already = md.plan_merge(source, [], SIGNS, categories={"greetings"})
    assert [p.row["sign_id"] for p in plans] == ["good_morning"]
    assert skipped == [] and already == 0


def test_no_categories_means_every_category():
    source = [row("S2", "circle", 1, "x"), row("S2", "good_morning", 1, "y", category="greetings")]
    plans, _, _ = md.plan_merge(source, [], SIGNS)
    assert len(plans) == 2
