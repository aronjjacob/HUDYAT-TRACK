"""import_public.plan_imports: FSL-105 files + public_mapping.csv -> PUB clips to create."""

import import_public as ip

MAPPING = [
    {"fsl105_id": "0", "fsl105_label": "GOOD MORNING", "category": "greetings", "sign_id": "good_morning"},
    {"fsl105_id": "7", "fsl105_label": "CIRCLE", "category": "shapes", "sign_id": "circle"},
]


def dests(plans):
    return [p.dest for p in plans]


def test_mapped_files_get_numbered_pub_destinations():
    plans, skipped = ip.plan_imports(MAPPING, ["0/b.MOV", "0/a.MOV", "7/x.mov"], [])
    assert dests(plans) == [
        "dataset/raw/greetings/good_morning/PUB/good_morning_PUB_01.mp4",
        "dataset/raw/greetings/good_morning/PUB/good_morning_PUB_02.mp4",
        "dataset/raw/shapes/circle/PUB/circle_PUB_01.mp4",
    ]
    assert plans[0].source == "0/a.MOV" and plans[0].take == 1
    assert skipped == []


def test_unmapped_labels_are_skipped_and_label_folders_match_too():
    plans, skipped = ip.plan_imports(MAPPING, ["Circle/1.MOV", "99/z.MOV", "THANK YOU/q.MOV"], [])
    assert dests(plans) == ["dataset/raw/shapes/circle/PUB/circle_PUB_01.mp4"]
    assert skipped == ["99/z.MOV", "THANK YOU/q.MOV"]


def test_rerun_skips_imported_files_and_continues_numbering():
    first, _ = ip.plan_imports(MAPPING, ["0/a.MOV", "0/b.MOV"], [])
    rows = [ip.metadata_row(p, "h264") for p in first]
    plans, _ = ip.plan_imports(MAPPING, ["0/a.MOV", "0/b.MOV", "0/c.MOV"], rows)
    assert [(p.source, p.take) for p in plans] == [("0/c.MOV", 3)]
    assert rows[0]["signer"] == "PUB" and rows[0]["file_path"] == first[0].dest
    assert rows[0]["status"] == "kept"
