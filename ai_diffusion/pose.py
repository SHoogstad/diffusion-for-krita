import math
import operator
import statistics
from functools import reduce
from typing import NamedTuple

from .image import Extent
from .qt.QtCore import QPointF
from .qt.QtGui import QTransform
from .util import batched


class Point(NamedTuple):
    x: float = 0
    y: float = 0

    @staticmethod
    def from_qt(qpoint: QPointF):
        return Point(qpoint.x(), qpoint.y())


body_parts = [
    "Nose",  # 0
    "Neck",  # 1
    "RShoulder",  # 2
    "RElbow",  # 3
    "RWrist",  # 4
    "LShoulder",  # 5
    "LElbow",  # 6
    "LWrist",  # 7
    "RHip",  # 8
    "RKnee",  # 9
    "RAnkle",  # 10
    "LHip",  # 11
    "LKnee",  # 12
    "LAnkle",  # 13
    "REye",  # 14
    "LEye",  # 15
    "REar",  # 16
    "LEar",  # 17
]
joint_count = len(body_parts)

default_positions = [
    Point(153, 107),  # Nose
    Point(152, 146),  # Neck
    Point(128, 146),  # RShoulder
    Point(116, 193),  # RElbow
    Point(100, 238),  # RWrist
    Point(176, 146),  # LShoulder
    Point(188, 193),  # LElbow
    Point(204, 238),  # LWrist
    Point(131, 227),  # RHip
    Point(129, 311),  # RKnee
    Point(128, 394),  # RAnkle
    Point(174, 227),  # LHip
    Point(176, 311),  # LKnee
    Point(178, 394),  # LAnkle
    Point(143, 100),  # REye
    Point(162, 100),  # LEye
    Point(132, 105),  # REar
    Point(174, 105),  # LEar
]

bone_connection = [
    (1, 2),  # 0
    (1, 5),  # 1
    (2, 3),  # 2
    (3, 4),  # 3
    (5, 6),  # 4
    (6, 7),  # 5
    (1, 8),  # 6
    (8, 9),  # 7
    (9, 10),  # 8
    (1, 11),  # 9
    (11, 12),  # 10
    (12, 13),  # 11
    (1, 0),  # 12
    (0, 14),  # 13
    (14, 16),  # 14
    (0, 15),  # 15
    (15, 17),  # 16
]
assert len(bone_connection) == joint_count - 1

colors = [
    "ff0000",
    "ff5500",
    "ffaa00",
    "ffff00",
    "aaff00",
    "55ff00",
    "00ff00",
    "00ff55",
    "00ffaa",
    "00ffff",
    "00aaff",
    "0055ff",
    "0000ff",
    "5500ff",
    "aa00ff",
    "ff00ff",
    "ff00aa",
    "ff0055",
]
assert len(body_parts) == joint_count


class JointIndex(NamedTuple):
    person: int
    joint: int

    @property
    def id(self):
        return f"P{self.person:02d}_J{self.joint:02d}"


class BoneIndex(NamedTuple):
    person: int
    bone: int

    @property
    def id(self):
        return f"P{self.person:02d}_B{self.bone:02d}"


def parse_id(string: str):
    if len(string) != 7 or string[0] != "P" or string[3] != "_":
        return None
    try:
        person, part = int(string[1:3]), int(string[5:7])
        if string[4] == "J":
            return JointIndex(person, part)
        elif string[4] == "B":
            return BoneIndex(person, part)
    except ValueError:
        pass  # can't parse number
    return None


def get_connected_bones(joint_id: int):
    return [i for i, (a, b) in enumerate(bone_connection) if joint_id in (a, b)]


def segment_transform(old: tuple[Point, Point], new: tuple[Point, Point]):
    """Transform which maps the line segment `old` onto `new`. It only stretches along the
    line, so the stroke width of a line shape stays the same."""
    (a0, b0), (a1, b1) = old, new
    l0, l1 = _distance(a0, b0), _distance(a1, b1)
    if l0 < 1e-6 or l1 < 1e-6:
        return None
    u0 = Point((b0.x - a0.x) / l0, (b0.y - a0.y) / l0)
    u1 = Point((b1.x - a1.x) / l1, (b1.y - a1.y) / l1)
    v0, v1 = Point(-u0.y, u0.x), Point(-u1.y, u1.x)
    k = l1 / l0
    # Linear part maps direction u0 to k*u1 and the perpendicular v0 to v1
    m11 = k * u1.x * u0.x + v1.x * v0.x
    m21 = k * u1.x * u0.y + v1.x * v0.y
    m12 = k * u1.y * u0.x + v1.y * v0.x
    m22 = k * u1.y * u0.y + v1.y * v0.y
    dx = a1.x - (m11 * a0.x + m21 * a0.y)
    dy = a1.y - (m12 * a0.x + m22 * a0.y)
    return QTransform(m11, m12, m21, m22, dx, dy)


class Shape:
    """Interface of the shapes passed to `Pose`, positions are in pixels.
    Also used as test mock, see `document.PoseShape` for the Krita implementation."""

    def __init__(self, name: str, position: Point):
        self._name = name
        self._position = QPointF(*position)
        self.removed = False
        self.line: tuple[Point, Point] | None = None

    def name(self):
        return self._name

    def setName(self, name):
        self._name = name

    def center(self):
        return Point.from_qt(self._position)

    def set_position(self, x, y):
        self._position = QPointF(x, y)

    def move_line(self, old: tuple[Point, Point], new: tuple[Point, Point]):
        """Move a bone (line) from its old to its new end points without re-creating it."""
        if segment_transform(old, new) is None:
            return False
        self.line = new
        return True

    def remove(self):
        self.removed = True


class Pose:
    people_count: int
    joints: dict[JointIndex, Point]

    _extent: Extent
    _stroke_width = 4.0
    _radius = 4.0

    def __init__(
        self,
        extent: Extent,
        people_count=0,
        joint_positions: dict[JointIndex, Point] | None = None,
    ):
        self.extent = extent
        self.people_count = people_count
        self.joints = joint_positions or {}

    @staticmethod
    def create_default(extent: Extent, person=0):
        s = extent.shortest_side / 400
        joints = {
            JointIndex(person, i): Point(p.x * s, p.y * s) for i, p in enumerate(default_positions)
        }
        return Pose(extent, 1, joints)

    @staticmethod
    def from_open_pose_json(pose: dict):
        # Format described at https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
        if isinstance(pose, list):
            pose = pose[0]  # Newer version of DWPose return a list of pose dicts
        extent = Extent(pose["canvas_width"], pose["canvas_height"])

        def parse_keypoints(person: int, keypoints: list[float]):
            assert len(keypoints) // 3 == joint_count, "Invalid keypoint count in OpenPose JSON"
            return {
                JointIndex(person, joint): Point(x, y)
                for joint, (x, y, confidence) in enumerate(batched(keypoints, 3))
                if confidence > 0.1
            }

        people = pose.get("people", [])
        poses = (parse_keypoints(i, p.get("pose_keypoints_2d", [])) for i, p in enumerate(people))
        return Pose(extent, len(people), reduce(operator.ior, poses, {}))

    @staticmethod
    def from_shapes(extent: Extent, shapes: list[Shape]):
        """Read joint positions from vector layer shapes without modifying them."""
        pose = Pose(extent)
        for shape in shapes:
            index = parse_id(shape.name())
            if isinstance(index, JointIndex) and index not in pose.joints:
                pose.joints[index] = shape.center()
                pose.people_count = max(pose.people_count, index.person + 1)
        return pose

    def scale(self, target: Extent):
        s = Point(target.width / self.extent.width, target.height / self.extent.height)
        self.joints = {i: Point(p.x * s.x, p.y * s.y) for i, p in self.joints.items()}
        self.extent = target

    def update(self, shapes: list[Shape]):
        """Sync with the joint shapes of a layer. Bones of moved joints are moved in place
        where possible (this doesn't create undo steps). Returns SVG with bones that have
        to be (re-)created, or None."""
        previous = dict(self.joints)
        moved = set()  # joints moved by the user
        added = set()  # joints seen for the first time, eg. after opening the document
        copied = set()  # joints which were duplicated, and their copies
        duplicates = set()
        bones: dict[BoneIndex, Shape] = {}
        new_people: dict[int, int] = {}

        for shape in shapes:
            index = parse_id(shape.name())
            if isinstance(index, JointIndex):
                pos = shape.center()
                previous_pos = self.joints.get(index)
                if previous_pos is None:
                    self.people_count = max(self.people_count, index.person + 1)
                    self.joints[index] = pos
                    added.add(index)
                elif index not in duplicates:
                    if pos != previous_pos:
                        self.joints[index] = pos
                        moved.add(index)
                else:  # shape with same id exists already, probably it was copied
                    person = new_people.setdefault(
                        index.person, self.people_count + len(new_people)
                    )
                    new_index = JointIndex(person, index.joint)
                    shape.setName(new_index.id)
                    self.joints[new_index] = pos
                    copied.update((index, new_index))
                duplicates.add(index)
            elif isinstance(index, BoneIndex):
                if duplicate := bones.get(index):
                    duplicate.remove()
                bones[index] = shape

        self.people_count += len(new_people)

        def connected_bones(joints: set[JointIndex]):
            return {
                BoneIndex(index.person, bone)
                for index in joints
                for bone in get_connected_bones(index.joint)
            }

        def end_points(joints: dict[JointIndex, Point], bone: BoneIndex):
            a, b = bone_connection[bone.bone]
            return joints.get(JointIndex(bone.person, a)), joints.get(JointIndex(bone.person, b))

        to_draw = connected_bones(copied)
        to_draw |= {b for b in connected_bones(added) if b not in bones}
        for bone in connected_bones(moved) - to_draw:
            shape = bones.get(bone)
            old, new = end_points(previous, bone), end_points(self.joints, bone)
            if shape and all(old) and all(new) and shape.move_line(old, new):  # type: ignore
                continue
            to_draw.add(bone)

        svg = ""
        for bone in to_draw:
            if shape := bones.get(bone):
                shape.remove()
            a, b = end_points(self.joints, bone)
            if a and b:
                svg += self._draw_bone(bone, a, b)
        if not svg:
            return None

        width, height = self.extent.width, self.extent.height
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
            f' viewBox="0 0 {width} {height}">{svg}</svg>'
        )

    def to_svg(self, bones=True):
        width, height = self.extent.width, self.extent.height
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
            f' viewBox="0 0 {width} {height}">'
        )

        for person in range(self.people_count if bones else 0):
            for i, bone in enumerate(bone_connection):
                beg = self.joints.get(JointIndex(person, bone[0]))
                end = self.joints.get(JointIndex(person, bone[1]))
                if beg and end:
                    svg += self._draw_bone(BoneIndex(person, i), beg, end)

        for index, position in self.joints.items():
            svg += self._draw_joint(index, position)

        return svg + "</svg>"

    @property
    def extent(self):
        return self._extent

    @extent.setter
    def extent(self, extent):
        self._extent = extent
        self._stroke_width = 4 + max(0, (self.extent.diagonal / 250) - 4)
        self._radius = self._stroke_width

    def _draw_bone(self, index: BoneIndex, a: Point, b: Point):
        return (
            f'<line id="{index.id}" x1="{a.x}" y1="{a.y}" x2="{b.x}" y2="{b.y}"'
            f' stroke="#{colors[index.bone]}" stroke-width="{self._stroke_width}"'
            ' stroke-opacity="0.6"/>'
        )

    def _draw_joint(self, index: JointIndex, pos: Point):
        return (
            f'<circle id="{index.id}" cx="{pos.x}" cy="{pos.y}" r="{self._radius}"'
            f' fill="#{colors[index.joint]}"/>'
        )


def _distance(a: Point, b: Point):
    return math.hypot(b.x - a.x, b.y - a.y)


def _bone_length(joints: dict[int, Point], bone: tuple[int, int]):
    a, b = joints.get(bone[0]), joints.get(bone[1])
    return _distance(a, b) if a and b else None


def _match_person_proportions(
    target: dict[int, Point], source: dict[int, Point], foreshortening: float
):
    # Overall size comes from the target, the median ignores outliers (foreshortened limbs)
    ratios = []
    for bone in bone_connection:
        source_len, target_len = _bone_length(source, bone), _bone_length(target, bone)
        if source_len and target_len:
            ratios.append(target_len / source_len)
    if not ratios:
        return dict(target)
    scale = statistics.median(ratios)

    # Walk the skeleton outwards from the neck: keep the target's bone directions and
    # take bone lengths from the source. Bones are ordered so parents come first.
    result = dict(target)
    for bone in bone_connection:
        parent, child = bone
        target_len = _bone_length(target, bone)
        if target_len is None:
            continue
        if target_len == 0:
            result[child] = result[parent]
            continue
        new_len = target_len
        if source_len := _bone_length(source, bone):
            new_len = source_len * scale
            # Bones much shorter than expected likely point towards the camera. Keep some
            # of that foreshortening instead of stretching them to full length.
            ratio = target_len / new_len
            if ratio < foreshortening:
                new_len *= ratio / foreshortening
        a, b, p = target[parent], target[child], result[parent]
        f = new_len / target_len
        result[child] = Point(p.x + (b.x - a.x) * f, p.y + (b.y - a.y) * f)
    return result


def match_proportions(target: Pose, source: Pose, foreshortening=0.5):
    """Returns a copy of the target pose with the body proportions (bone lengths) of the
    source pose. Bone directions, neck position and overall size of the target are kept.
    Source person N is applied to target person N, or person 0 if there is no person N."""

    def person_joints(pose: Pose, person: int):
        return {i.joint: p for i, p in pose.joints.items() if i.person == person}

    joints: dict[JointIndex, Point] = {}
    for person in range(target.people_count):
        source_joints = person_joints(source, person) or person_joints(source, 0)
        matched = _match_person_proportions(
            person_joints(target, person), source_joints, foreshortening
        )
        joints.update({JointIndex(person, j): p for j, p in matched.items()})
    return Pose(target.extent, target.people_count, joints)
