import math

import pytest

from ai_diffusion.image import Extent
from ai_diffusion.pose import (
    BoneIndex,
    JointIndex,
    Point,
    Pose,
    Shape,
    bone_connection,
    default_positions,
    get_connected_bones,
    match_proportions,
    parse_id,
    segment_transform,
)
from ai_diffusion.qt.QtCore import QPointF
from ai_diffusion.qt.QtGui import QTransform


def test_bone_id():
    assert BoneIndex(0, 0).id == "P00_B00"
    assert BoneIndex(0, 1).id == "P00_B01"
    assert BoneIndex(1, 0).id == "P01_B00"
    assert BoneIndex(1, 1).id == "P01_B01"


def test_joint_id():
    assert JointIndex(0, 0).id == "P00_J00"
    assert JointIndex(0, 1).id == "P00_J01"
    assert JointIndex(1, 0).id == "P01_J00"
    assert JointIndex(1, 1).id == "P01_J01"


def test_parse_id():
    assert parse_id("P00_J00") == JointIndex(0, 0)
    assert parse_id("P00_J01") == JointIndex(0, 1)
    assert parse_id("P01_J00") == JointIndex(1, 0)
    assert parse_id("P01_J01") == JointIndex(1, 1)
    assert parse_id("P00_B00") == BoneIndex(0, 0)
    assert parse_id("P00_B01") == BoneIndex(0, 1)
    assert parse_id("P01_B00") == BoneIndex(1, 0)
    assert parse_id("P01_B01") == BoneIndex(1, 1)
    assert parse_id("") is None
    assert parse_id("P00") is None
    assert parse_id("P00_X82") is None
    assert parse_id("P00_J82_bla") is None


def test_connected_bones():
    assert get_connected_bones(0) == [12, 13, 15]
    assert get_connected_bones(1) == [0, 1, 6, 9, 12]
    assert get_connected_bones(2) == [0, 2]
    assert get_connected_bones(3) == [2, 3]
    assert get_connected_bones(4) == [3]


def test_pose_from_json():
    json = {
        "canvas_width": 123,
        "canvas_height": 456,
        "people": [
            {
                "pose_keypoints_2d": [11, 12, 1, 21, 22, 1, 31, 32, 0.5, 41, 42, 0.9, 51, 52, 0]
                + [0 for i in range(3 * 13)]
            },
            {
                "pose_keypoints_2d": [-11, -12, 1, -21, -22, 1, -31, -32, 0.5]
                + [0 for i in range(3 * 15)]
            },
        ],
    }
    pose = Pose.from_open_pose_json(json)
    assert pose.people_count == 2
    assert pose.extent == Extent(123, 456)
    assert len(pose.joints) == 4 + 3
    assert pose.joints[JointIndex(0, 0)] == Point(11, 12)
    assert pose.joints[JointIndex(0, 1)] == Point(21, 22)
    assert pose.joints[JointIndex(0, 2)] == Point(31, 32)
    assert pose.joints[JointIndex(0, 3)] == Point(41, 42)
    assert pose.joints[JointIndex(1, 0)] == Point(-11, -12)


def test_pose_to_svg():
    joints = {
        JointIndex(0, 0): Point(11, 12),
        JointIndex(0, 1): Point(21, 22),
        JointIndex(0, 14): Point(31, 32),
    }
    pose = Pose(Extent(123, 456), 1, joints)
    svg = pose.to_svg()
    assert '<circle id="P00_J00" cx="11" cy="12"' in svg
    assert '<circle id="P00_J01" cx="21" cy="22"' in svg
    assert '<circle id="P00_J14" cx="31" cy="32"' in svg
    assert '<line id="P00_B12" x1="21" y1="22" x2="11" y2="12"' in svg
    assert '<line id="P00_B13" x1="11" y1="12" x2="31" y2="32"' in svg


def _bone_line(shape: Shape, drawn_at: tuple[Point, Point]):
    """Where a mock bone line drawn at `drawn_at` is now, after transformations."""
    t = shape.layer_transform()
    return tuple(Point.from_qt(t.map(QPointF(*p))) for p in drawn_at)


def _approx_line(line: tuple[Point, ...], expected: tuple[Point, Point]):
    return all(
        a.x == pytest.approx(b.x, abs=1e-6) and a.y == pytest.approx(b.y, abs=1e-6)
        for a, b in zip(line, expected)
    )


def _pose_shapes():
    # Bone 12 connects neck (J01) and nose (J00), bone 13 nose and right eye (J14)
    return [
        Shape("P00_J00", Point(11, 12)),
        Shape("P00_J01", Point(21, 22)),
        Shape("P00_J14", Point(31, 32)),
        Shape("P00_B12", Point(21, 22)),
        Shape("P00_B13", Point(11, 12)),
    ]


B12 = (Point(21, 22), Point(11, 12))
B13 = (Point(11, 12), Point(31, 32))


def test_pose_update():
    pose = Pose(Extent(123, 456))
    shapes = _pose_shapes()
    changes = pose.update(shapes)
    assert changes is None  # bones exist already, nothing to redraw
    assert pose.people_count == 1
    assert len(pose.joints) == 3
    assert pose.joints[JointIndex(0, 0)] == Point(11, 12)

    # Bones of moved joints are moved in place
    shapes[2].set_position(31, 38)
    changes = pose.update(shapes)
    assert changes is None
    assert _approx_line(_bone_line(shapes[3], B12), B12)
    assert _approx_line(_bone_line(shapes[4], B13), (Point(11, 12), Point(31, 38)))
    assert not any(s.removed for s in shapes)

    shapes[0].set_position(11, 18)
    changes = pose.update(shapes)
    assert changes is None
    assert _approx_line(_bone_line(shapes[3], B12), (Point(21, 22), Point(11, 18)))
    assert _approx_line(_bone_line(shapes[4], B13), (Point(11, 18), Point(31, 38)))
    assert not any(s.removed for s in shapes)

    # Unchanged positions don't do anything
    transforms = [s.layer_transform() for s in shapes]
    assert pose.update(shapes) is None
    assert [s.layer_transform() for s in shapes] == transforms

    # Undo: joint goes back, bones follow
    shapes[0].set_position(11, 12)
    shapes[2].set_position(31, 32)
    assert pose.update(shapes) is None
    assert _approx_line(_bone_line(shapes[3], B12), B12)
    assert _approx_line(_bone_line(shapes[4], B13), B13)


def test_pose_update_bones_moved_with_joints():
    # User selects joints and bones and drags them all by (100, 50)
    pose = Pose(Extent(500, 500))
    shapes = _pose_shapes()
    pose.update(shapes)
    for s in shapes[:3]:
        c = s.center()
        s.set_position(c.x + 100, c.y + 50)
    for s in shapes[3:]:
        s.set_layer_transform(s.layer_transform() * QTransform.fromTranslate(100, 50))

    assert pose.update(shapes) is None
    moved_b12 = (Point(121, 72), Point(111, 62))
    moved_b13 = (Point(111, 62), Point(131, 82))
    assert _approx_line(_bone_line(shapes[3], B12), moved_b12)  # not moved twice
    assert _approx_line(_bone_line(shapes[4], B13), moved_b13)


def test_pose_update_bone_moved_alone():
    # A bone line dragged away by itself snaps back to its joints
    pose = Pose(Extent(500, 500))
    shapes = _pose_shapes()
    pose.update(shapes)
    shapes[3].set_layer_transform(QTransform.fromTranslate(40, -5))
    assert pose.update(shapes) is None
    assert _approx_line(_bone_line(shapes[3], B12), B12)


def test_pose_update_redraw():
    pose = Pose(Extent(123, 456))
    shapes = _pose_shapes()[:4]
    # Bone 13 is missing, it is drawn when the joints are first seen
    changes = pose.update(shapes)
    assert changes is not None
    assert "B12" not in changes
    assert '<line id="P00_B13" x1="11.0" y1="12.0" x2="31.0" y2="32.0"' in changes

    # A bone deleted by the user is drawn again when one of its joints moves
    assert pose.update(shapes) is None  # not redrawn on every update
    shapes[2].set_position(31, 33)
    changes = pose.update(shapes)
    assert changes is not None
    assert '<line id="P00_B13" x1="11.0" y1="12.0" x2="31.0" y2="33.0"' in changes

    # Collapsing a bone to zero length can't be done with a transform, it's redrawn
    shapes[0].set_position(21, 22)
    changes = pose.update(shapes)
    assert changes is not None
    assert '<line id="P00_B12" x1="21.0" y1="22.0" x2="21.0" y2="22.0"' in changes
    assert shapes[3].removed


def test_segment_transform():
    old = (Point(10, 10), Point(20, 10))
    new = (Point(50, 50), Point(50, 80))
    m = segment_transform(old, new)
    assert m is not None
    for a, b in zip(old, new):
        p = m.map(QPointF(a.x, a.y))
        assert p.x() == pytest.approx(b.x) and p.y() == pytest.approx(b.y)
    # Perpendicular direction is rotated but not scaled (stroke width stays the same)
    p0, p1 = m.map(QPointF(10, 10)), m.map(QPointF(10, 14))
    assert math.hypot(p1.x() - p0.x(), p1.y() - p0.y()) == pytest.approx(4)

    assert segment_transform((Point(1, 1), Point(1, 1)), new) is None
    assert segment_transform(old, (Point(1, 1), Point(1, 1))) is None


@pytest.mark.parametrize("scenario", ["with_bones", "without_bones"])
def test_pose_update_copy(scenario):
    pose = Pose(Extent(123, 456))
    shapes = [
        Shape("P00_J00", Point(11, 12)),
        Shape("P00_J01", Point(21, 22)),
    ]
    pose.update(shapes)
    assert pose.people_count == 1
    assert len(pose.joints) == 2

    shapes[1].set_position(21, 32)
    shapes += [
        Shape("P00_J00", Point(51, 52)),
        Shape("P00_J01", Point(61, 62)),
    ]
    if scenario == "with_bones":
        shapes += [
            Shape("P00_B12", Point(21, 22)),
            Shape("P00_B12", Point(21, 22)),
        ]
    changes = pose.update(shapes)
    assert pose.people_count == 2
    assert pose.joints[JointIndex(0, 0)] == Point(11, 12)
    assert pose.joints[JointIndex(0, 1)] == Point(21, 32)
    assert pose.joints[JointIndex(1, 0)] == Point(51, 52)
    assert pose.joints[JointIndex(1, 1)] == Point(61, 62)
    assert changes is not None
    assert '<line id="P00_B12" x1="21.0" y1="32.0" x2="11.0" y2="12.0"' in changes
    assert '<line id="P01_B12" x1="61.0" y1="62.0" x2="51.0" y2="52.0"' in changes
    assert shapes[2].name() == "P01_J00"
    assert shapes[3].name() == "P01_J01"
    if scenario == "with_bones":
        assert shapes[-1].removed and shapes[-2].removed


def test_stroke_width():
    joints = {JointIndex(0, 0): Point(100, 100), JointIndex(0, 1): Point(200, 200)}
    pose = Pose(Extent(2000, 2000), 1, joints)
    svg = pose.to_svg()
    assert 'stroke-width="11.' in svg
    assert 'r="11.' in svg


def _person(positions: list[Point] | dict[int, Point], person=0):
    items = enumerate(positions) if isinstance(positions, list) else positions.items()
    return {JointIndex(person, i): p for i, p in items}


def _length(joints: dict[JointIndex, Point], bone: int, person=0):
    a, b = bone_connection[bone]
    pa, pb = joints[JointIndex(person, a)], joints[JointIndex(person, b)]
    return math.hypot(pb.x - pa.x, pb.y - pa.y)


def _direction(joints: dict[JointIndex, Point], bone: int, person=0):
    a, b = bone_connection[bone]
    pa, pb = joints[JointIndex(person, a)], joints[JointIndex(person, b)]
    return math.atan2(pb.y - pa.y, pb.x - pa.x)


def _scaled(positions: list[Point], s: float, offset: Point | None = None):
    offset = offset or Point()
    return [Point(p.x * s + offset.x, p.y * s + offset.y) for p in positions]


def _long_legs(positions: list[Point]):
    # Stretch thighs and shins of the default skeleton (bones 7, 8, 10, 11)
    result = list(positions)
    for hip, knee, ankle in ((8, 9, 10), (11, 12, 13)):
        h, k, a = positions[hip], positions[knee], positions[ankle]
        k2 = Point(h.x + (k.x - h.x) * 1.5, h.y + (k.y - h.y) * 1.5)
        result[knee] = k2
        result[ankle] = Point(k2.x + (a.x - k.x) * 1.5, k2.y + (a.y - k.y) * 1.5)
    return result


def test_match_proportions_identical():
    joints = _person(default_positions)
    target = Pose(Extent(400, 400), 1, joints)
    result = match_proportions(target, Pose(Extent(400, 400), 1, dict(joints)))
    for index, p in joints.items():
        assert result.joints[index].x == pytest.approx(p.x)
        assert result.joints[index].y == pytest.approx(p.y)


def test_match_proportions_keeps_target_directions_and_size():
    # Target: default skeleton, twice as big and moved. Source: long legs, original size.
    offset = Point(300, 50)
    target = Pose(Extent(1000, 1000), 1, _person(_scaled(default_positions, 2, offset)))
    source = Pose(Extent(400, 400), 1, _person(_long_legs(default_positions)))
    result = match_proportions(target, source)

    neck = JointIndex(0, 1)
    assert result.joints[neck] == target.joints[neck]
    assert result.people_count == 1 and len(result.joints) == len(target.joints)
    for bone in range(len(bone_connection)):
        assert _direction(result.joints, bone) == pytest.approx(_direction(target.joints, bone))
        source_ratio = _length(source.joints, bone) / _length(source.joints, 2)  # vs shoulder
        result_ratio = _length(result.joints, bone) / _length(result.joints, 2)
        assert result_ratio == pytest.approx(source_ratio)
    # Most bones are unchanged, so the size stays that of the target
    assert _length(result.joints, 2) == pytest.approx(_length(target.joints, 2))
    assert _length(result.joints, 7) == pytest.approx(_length(target.joints, 7) * 1.5)


def test_match_proportions_foreshortened():
    positions = list(default_positions)
    elbow, wrist = positions[3], positions[4]
    # Right forearm points at the camera: only 30% of its length is visible
    positions[4] = Point(elbow.x + (wrist.x - elbow.x) * 0.3, elbow.y + (wrist.y - elbow.y) * 0.3)
    target = Pose(Extent(400, 400), 1, _person(positions))
    source = Pose(Extent(400, 400), 1, _person(default_positions))

    result = match_proportions(target, source)
    full = _length(source.joints, 3)
    assert _length(result.joints, 3) == pytest.approx(full * 0.3 / 0.5)

    result = match_proportions(target, source, foreshortening=0)
    assert _length(result.joints, 3) == pytest.approx(full)


def test_match_proportions_missing_joints():
    target_positions = dict(enumerate(default_positions))
    del target_positions[3]  # right elbow
    target = Pose(Extent(400, 400), 1, _person(target_positions))
    source_positions = dict(enumerate(_long_legs(default_positions)))
    del source_positions[9]  # right knee
    source = Pose(Extent(400, 400), 1, _person(source_positions))

    result = match_proportions(target, source)
    assert JointIndex(0, 3) not in result.joints
    # Wrist has no parent in the target, it stays where it was
    assert result.joints[JointIndex(0, 4)] == target.joints[JointIndex(0, 4)]
    # Right leg has no source length, keeps the target's
    assert _length(result.joints, 7) == pytest.approx(_length(target.joints, 7))
    assert _length(result.joints, 10) == pytest.approx(_length(target.joints, 10) * 1.5)


def test_match_proportions_people():
    target_joints = _person(default_positions, 0)
    target_joints.update(_person(_scaled(default_positions, 1, Point(200, 0)), 1))
    target_joints.update(_person(_scaled(default_positions, 1, Point(400, 0)), 2))
    target = Pose(Extent(800, 400), 3, target_joints)
    source_joints = _person(_long_legs(default_positions), 0)
    source_joints.update(_person(default_positions, 1))
    source = Pose(Extent(400, 400), 2, source_joints)

    result = match_proportions(target, source)
    assert result.people_count == 3
    leg = _length(target.joints, 7)
    assert _length(result.joints, 7, person=0) == pytest.approx(leg * 1.5)
    assert _length(result.joints, 7, person=1) == pytest.approx(leg)
    assert _length(result.joints, 7, person=2) == pytest.approx(leg * 1.5)  # uses person 0


def test_match_proportions_empty_source():
    target = Pose(Extent(400, 400), 1, _person(default_positions))
    result = match_proportions(target, Pose(Extent(400, 400)))
    assert result.joints == target.joints


def test_pose_from_shapes():
    shapes = [
        Shape("P00_J00", Point(11, 12)),
        Shape("P00_J01", Point(21, 22)),
        Shape("P00_B12", Point(21, 22)),
        Shape("P01_J01", Point(31, 32)),
        Shape("P01_J01", Point(41, 42)),  # duplicate, first one wins
        Shape("Not a joint", Point(1, 2)),
    ]
    pose = Pose.from_shapes(Extent(123, 456), shapes)
    assert pose.people_count == 2
    assert pose.joints == {
        JointIndex(0, 0): Point(11, 12),
        JointIndex(0, 1): Point(21, 22),
        JointIndex(1, 1): Point(31, 32),
    }
    assert not any(s.removed for s in shapes)
    assert [s.name() for s in shapes][3:5] == ["P01_J01", "P01_J01"]


def test_pose_to_svg_without_bones():
    joints = {JointIndex(0, 0): Point(11, 12), JointIndex(0, 1): Point(21, 22)}
    svg = Pose(Extent(123, 456), 1, joints).to_svg(bones=False)
    assert "<circle" in svg and "<line" not in svg


def test_pose_update_deleted_joint():
    # Deleting a joint and its bones leaves that body part out for good
    pose = Pose(Extent(123, 456))
    shapes = _pose_shapes()
    pose.update(shapes)
    shapes = [s for s in shapes if s.name() not in ("P00_J14", "P00_B13")]
    assert pose.update(shapes) is None
    assert JointIndex(0, 14) not in pose.joints

    shapes[0].set_position(11, 18)  # nose, connected to the deleted bone 13
    assert pose.update(shapes) is None
    assert _approx_line(_bone_line(shapes[2], B12), (Point(21, 22), Point(11, 18)))
