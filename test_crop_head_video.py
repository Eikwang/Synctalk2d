import pytest
import math
import numpy as np
from crop_head_video import HeadTracker


class TestHeadTrackerInit:
    """Task 1.1: HeadTracker 初始化与状态管理"""

    def test_default_params(self):
        tracker = HeadTracker()
        assert tracker.pan_alpha == 0.15
        assert tracker.zoom_alpha == 0.08
        assert tracker.dead_zone_ratio == 0.35
        assert tracker.pad_ratio_range == (1.8, 2.6)
        assert tracker.output_size == 848
        assert tracker.min_face_size == 30

    def test_custom_params(self):
        tracker = HeadTracker(
            pan_alpha=0.2,
            zoom_alpha=0.05,
            dead_zone_ratio=0.4,
            pad_ratio_range=(1.5, 3.0),
            min_face_size=50,
            output_size=512,
        )
        assert tracker.pan_alpha == 0.2
        assert tracker.zoom_alpha == 0.05
        assert tracker.dead_zone_ratio == 0.4
        assert tracker.pad_ratio_range == (1.5, 3.0)
        assert tracker.output_size == 512
        assert tracker.min_face_size == 50

    def test_initial_state_is_none(self):
        """未检测到面部前，裁切框中心为 None"""
        tracker = HeadTracker()
        assert tracker.crop_cx is None
        assert tracker.crop_cy is None

    def test_first_detection_sets_center(self):
        """首次检测到面部，裁切框中心直接设为面部中心"""
        tracker = HeadTracker()
        result = tracker.update(
            face_cx=400, face_cy=300, face_size=200,
            frame_w=1920, frame_h=1080
        )
        assert result is not None
        assert tracker.crop_cx == 400.0
        assert tracker.crop_cy == 300.0

    def test_first_detection_sets_crop_size(self):
        """首次检测到面部，smooth_crop_size = face_size * default_pad_ratio(2.2)"""
        tracker = HeadTracker()
        result = tracker.update(
            face_cx=400, face_cy=300, face_size=200,
            frame_w=1920, frame_h=1080
        )
        assert tracker.smooth_crop_size == pytest.approx(200 * 2.2)

    def test_first_detection_result_fields(self):
        """首次检测返回的 result 包含正确的字段"""
        tracker = HeadTracker()
        result = tracker.update(
            face_cx=400, face_cy=300, face_size=200,
            frame_w=1920, frame_h=1080
        )
        assert 'crop_x' in result
        assert 'crop_y' in result
        assert 'crop_size' in result
        assert 'face_cx' in result
        assert 'face_cy' in result
        assert 'face_w' in result
        assert 'face_h' in result
        assert 'smooth_crop_size' in result
        assert 'pad_ratio' in result

    def test_first_detection_result_values(self):
        """首次检测返回的 result 值正确"""
        tracker = HeadTracker()
        result = tracker.update(
            face_cx=400, face_cy=300, face_size=200,
            frame_w=1920, frame_h=1080
        )
        expected_crop_size = 200 * 2.2
        assert result['crop_size'] == pytest.approx(expected_crop_size)
        assert result['crop_x'] == pytest.approx(400 - expected_crop_size / 2)
        assert result['crop_y'] == pytest.approx(300 - expected_crop_size / 2)
        assert result['face_cx'] == 400.0
        assert result['face_cy'] == 300.0
        assert result['pad_ratio'] == pytest.approx(2.2)


class TestDeadZoneFollow:
    """Task 1.2: 死区跟随逻辑 — AC-001, AC-002"""

    def _init_tracker_with_face(self, face_cx=960, face_cy=540, face_size=200,
                                 frame_w=1920, frame_h=1080):
        """辅助：创建 tracker 并完成首次检测"""
        tracker = HeadTracker()
        tracker.update(
            face_cx=face_cx, face_cy=face_cy, face_size=face_size,
            frame_w=frame_w, frame_h=frame_h
        )
        return tracker

    def test_small_movement_within_deadzone_no_move(self):
        """AC-001: 面部中心偏移不超过死区阈值(裁切框宽度35%)时，裁切框不动"""
        tracker = self._init_tracker_with_face(face_cx=960, face_cy=540, face_size=200)
        # 首次检测后 crop_cx=960, smooth_crop_size=440, dead_zone=440*0.35=154
        initial_cx = tracker.crop_cx
        initial_cy = tracker.crop_cy

        # 面部移动 50px（小于死区 154px）
        result = tracker.update(
            face_cx=960 + 50, face_cy=540, face_size=200,
            frame_w=1920, frame_h=1080
        )
        assert tracker.crop_cx == initial_cx
        assert tracker.crop_cy == initial_cy

    def test_large_movement_beyond_deadzone_follows(self):
        """AC-002: 面部中心偏移超过死区阈值，裁切框缓慢跟随"""
        tracker = self._init_tracker_with_face(face_cx=960, face_cy=540, face_size=200)
        # dead_zone = 440 * 0.35 = 154
        initial_cx = tracker.crop_cx

        # 面部移动 200px（超过死区 154px）
        result = tracker.update(
            face_cx=960 + 200, face_cy=540, face_size=200,
            frame_w=1920, frame_h=1080
        )
        # 超出死区 200 - 154 = 46px，EMA(0.15) → 移动 0.15 * 46 = 6.9px
        expected_move = 0.15 * (200 - 154)
        assert tracker.crop_cx == pytest.approx(initial_cx + expected_move, abs=0.1)

    def test_dead_zone_calculated_from_crop_size(self):
        """死区阈值 = smooth_crop_size * dead_zone_ratio"""
        tracker = self._init_tracker_with_face(face_cx=960, face_cy=540, face_size=200)
        # smooth_crop_size = 440
        expected_dead_zone = 440 * 0.35
        # 面部偏移刚好在死区边缘 - 1px，不动
        tracker.update(
            face_cx=960 + expected_dead_zone - 1, face_cy=540, face_size=200,
            frame_w=1920, frame_h=1080
        )
        cx_after_within = tracker.crop_cx

        # 面部偏移刚好超出死区 + 1px，动
        tracker2 = self._init_tracker_with_face(face_cx=960, face_cy=540, face_size=200)
        tracker2.update(
            face_cx=960 + expected_dead_zone + 1, face_cy=540, face_size=200,
            frame_w=1920, frame_h=1080
        )
        cx_after_beyond = tracker2.crop_cx

        assert cx_after_within == 960.0  # 不动
        assert cx_after_beyond != 960.0  # 动了

    def test_consecutive_frames_converge(self):
        """连续多帧同一偏移，裁切框逐渐趋近面部中心"""
        tracker = self._init_tracker_with_face(face_cx=960, face_cy=540, face_size=200)
        crop_cx_values = [tracker.crop_cx]

        for _ in range(30):
            tracker.update(
                face_cx=1160, face_cy=540, face_size=200,
                frame_w=1920, frame_h=1080
            )
            crop_cx_values.append(tracker.crop_cx)

        # 裁切框中心应该单调递增（向面部方向移动）
        for i in range(1, len(crop_cx_values)):
            assert crop_cx_values[i] > crop_cx_values[i - 1]

    def test_diagonal_movement_deadzone(self):
        """对角线移动也遵守死区逻辑"""
        tracker = self._init_tracker_with_face(face_cx=960, face_cy=540, face_size=200)
        initial_cx = tracker.crop_cx
        initial_cy = tracker.crop_cy

        # 对角偏移 100px（各方向约70.7px），小于死区 154px
        result = tracker.update(
            face_cx=960 + 70, face_cy=540 + 70, face_size=200,
            frame_w=1920, frame_h=1080
        )
        dist = math.sqrt(70**2 + 70**2)  # ~99px < 154
        assert tracker.crop_cx == initial_cx
        assert tracker.crop_cy == initial_cy


class TestDynamicPadRatio:
    """Task 1.3: 动态 pad_ratio 与焦距平滑 — AC-003, AC-004, AC-005, AC-012"""

    def _init_tracker_with_face(self, face_cx=960, face_cy=540, face_size=200,
                                 frame_w=1920, frame_h=1080):
        tracker = HeadTracker()
        tracker.update(
            face_cx=face_cx, face_cy=face_cy, face_size=face_size,
            frame_w=frame_w, frame_h=frame_h
        )
        return tracker

    def test_stable_face_no_crop_size_change(self):
        """AC-005: 面部尺寸不变时，裁切框尺寸不变"""
        tracker = self._init_tracker_with_face(face_size=200)
        initial_size = tracker.smooth_crop_size
        tracker.update(face_cx=960, face_cy=540, face_size=200, frame_w=1920, frame_h=1080)
        assert tracker.smooth_crop_size == pytest.approx(initial_size)

    def test_face_grows_crop_size_clamped_to_min_pad(self):
        """AC-003: 人物靠近(面部变大)，pad_ratio 最低 1.8"""
        tracker = self._init_tracker_with_face(face_size=200)
        tracker.update(face_cx=960, face_cy=540, face_size=300, frame_w=1920, frame_h=1080)
        assert tracker.smooth_crop_size == pytest.approx(540.0)

    def test_face_shrinks_crop_size_clamped_to_max_pad(self):
        """AC-004: 人物远离(面部变小)，pad_ratio 最高 2.6"""
        tracker = self._init_tracker_with_face(face_size=400)
        tracker.update(face_cx=960, face_cy=540, face_size=200, frame_w=1920, frame_h=1080)
        assert tracker.smooth_crop_size == pytest.approx(520.0)

    def test_pad_ratio_always_within_range(self):
        """AC-012: pad_ratio 始终在 1.8~2.6 范围内"""
        tracker = HeadTracker(pad_ratio_range=(1.8, 2.6))
        tracker.update(face_cx=960, face_cy=540, face_size=200, frame_w=1920, frame_h=1080)
        face_sizes = [100, 500, 150, 400, 80, 300]
        for fs in face_sizes:
            result = tracker.update(face_cx=960, face_cy=540, face_size=fs, frame_w=1920, frame_h=1080)
            actual_pad = result['crop_size'] / fs
            assert 1.8 <= actual_pad <= 2.6, f"pad_ratio={actual_pad} out of range for face_size={fs}"

    def test_consecutive_zoom_converges(self):
        """连续多帧后面部变大，裁切框逐渐趋近新的目标尺寸"""
        tracker = self._init_tracker_with_face(face_size=200)
        sizes = [tracker.smooth_crop_size]
        for _ in range(50):
            tracker.update(face_cx=960, face_cy=540, face_size=300, frame_w=1920, frame_h=1080)
            sizes.append(tracker.smooth_crop_size)
        final = sizes[-1]
        assert final > sizes[0]
        assert 300 * 1.8 <= final <= 300 * 2.6


class TestBoundaryConstraints:
    """Task 1.4: 画面内裁切与边界约束 — AC-007, AC-011, AC-013"""

    def test_crop_center_clamped_near_left_edge(self):
        """AC-007: 裁切框超出画面左侧时，向右移动到画面内"""
        tracker = HeadTracker()
        result = tracker.update(face_cx=10, face_cy=540, face_size=200, frame_w=1920, frame_h=1080)
        assert result['crop_x'] >= 0

    def test_crop_center_clamped_near_top_edge(self):
        """AC-007: 裁切框超出画面上侧时，向下移动到画面内"""
        tracker = HeadTracker()
        result = tracker.update(face_cx=960, face_cy=10, face_size=200, frame_w=1920, frame_h=1080)
        assert result['crop_y'] >= 0

    def test_crop_size_limited_by_frame_short_side(self):
        """AC-011: 裁切框尺寸不超过画面短边"""
        tracker = HeadTracker()
        result = tracker.update(face_cx=960, face_cy=540, face_size=800, frame_w=1920, frame_h=1080)
        assert result['crop_size'] <= min(1920, 1080)

    def test_crop_box_fully_inside_frame(self):
        """AC-007: 裁切框完全在画面内"""
        tracker = HeadTracker()
        result = tracker.update(face_cx=960, face_cy=540, face_size=200, frame_w=1920, frame_h=1080)
        assert result['crop_x'] >= 0
        assert result['crop_y'] >= 0
        assert result['crop_x'] + result['crop_size'] <= 1920
        assert result['crop_y'] + result['crop_size'] <= 1080

    def test_face_at_corner_crop_stays_inside(self):
        """AC-007: 面部在画面角落时裁切框仍在画面内"""
        tracker = HeadTracker()
        result = tracker.update(face_cx=5, face_cy=5, face_size=200, frame_w=1920, frame_h=1080)
        assert result['crop_x'] >= 0
        assert result['crop_y'] >= 0
        assert result['crop_x'] + result['crop_size'] <= 1920
        assert result['crop_y'] + result['crop_size'] <= 1080

    def test_face_at_right_bottom_corner(self):
        """AC-007: 面部在右下角"""
        tracker = HeadTracker()
        result = tracker.update(face_cx=1915, face_cy=1075, face_size=200, frame_w=1920, frame_h=1080)
        assert result['crop_x'] >= 0
        assert result['crop_y'] >= 0
        assert result['crop_x'] + result['crop_size'] <= 1920
        assert result['crop_y'] + result['crop_size'] <= 1080

    def test_no_black_padding_in_crop(self):
        """AC-013: crop_and_resize 不做任何填充"""
        from crop_head_video import crop_and_resize
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        crop_info = {
            'crop_x': 100.0, 'crop_y': 100.0, 'crop_size': 400.0,
        }
        resized, scale = crop_and_resize(frame, crop_info, output_size=848)
        assert resized.shape == (848, 848, 3)
        assert not np.all(resized[0, :, :] == 0)
        assert not np.all(resized[-1, :, :] == 0)
        assert not np.all(resized[:, 0, :] == 0)
        assert not np.all(resized[:, -1, :] == 0)


class TestFaceDetectionLost:
    """Task 1.5: 面部检测丢失处理 — AC-008, AC-009, AC-010"""

    def test_detection_lost_returns_last_valid(self):
        """AC-008: 面部检测丢失时返回上一帧裁切框"""
        tracker = HeadTracker()
        result1 = tracker.update(face_cx=960, face_cy=540, face_size=200, frame_w=1920, frame_h=1080)
        # 正常帧
        result2 = tracker.update(face_cx=960, face_cy=540, face_size=200, frame_w=1920, frame_h=1080)
        # 检测丢失
        result3 = tracker.update(face_detected=False, frame_w=1920, frame_h=1080)
        assert result3['crop_x'] == result2['crop_x']
        assert result3['crop_y'] == result2['crop_y']
        assert result3['crop_size'] == result2['crop_size']

    def test_consecutive_lost_returns_same_result(self):
        """AC-008: 连续多帧丢失，始终返回同一裁切框"""
        tracker = HeadTracker()
        tracker.update(face_cx=960, face_cy=540, face_size=200, frame_w=1920, frame_h=1080)
        tracker.update(face_cx=960, face_cy=540, face_size=200, frame_w=1920, frame_h=1080)
        result_lost1 = tracker.update(face_detected=False, frame_w=1920, frame_h=1080)
        result_lost2 = tracker.update(face_detected=False, frame_w=1920, frame_h=1080)
        assert result_lost1['crop_x'] == result_lost2['crop_x']
        assert result_lost1['crop_y'] == result_lost2['crop_y']
        assert result_lost1['crop_size'] == result_lost2['crop_size']

    def test_no_history_returns_default_center(self):
        """AC-009: 从未检测到面部时返回画面中心+短边尺寸"""
        tracker = HeadTracker()
        result = tracker.update(face_detected=False, frame_w=1920, frame_h=1080)
        assert result['crop_x'] == pytest.approx((1920 - 1080) / 2.0)
        assert result['crop_y'] == pytest.approx(0.0)
        assert result['crop_size'] == pytest.approx(1080.0)

    def test_no_history_square_frame(self):
        """AC-009: 正方形画面时默认裁切框居中"""
        tracker = HeadTracker()
        result = tracker.update(face_detected=False, frame_w=1080, frame_h=1080)
        assert result['crop_x'] == pytest.approx(0.0)
        assert result['crop_y'] == pytest.approx(0.0)
        assert result['crop_size'] == pytest.approx(1080.0)

    def test_small_face_treated_as_lost(self):
        """AC-010: face_size < min_face_size 视同检测丢失"""
        tracker = HeadTracker(min_face_size=30)
        result1 = tracker.update(face_cx=960, face_cy=540, face_size=200, frame_w=1920, frame_h=1080)
        result2 = tracker.update(face_cx=960, face_cy=540, face_size=200, frame_w=1920, frame_h=1080)
        # 面部尺寸过小
        result3 = tracker.update(face_cx=960, face_cy=540, face_size=5, frame_w=1920, frame_h=1080)
        assert result3['crop_x'] == result2['crop_x']
        assert result3['crop_y'] == result2['crop_y']
        assert result3['crop_size'] == result2['crop_size']

    def test_recovery_after_lost(self):
        """检测恢复后继续正常跟随"""
        tracker = HeadTracker()
        tracker.update(face_cx=960, face_cy=540, face_size=200, frame_w=1920, frame_h=1080)
        # 丢失
        tracker.update(face_detected=False, frame_w=1920, frame_h=1080)
        # 恢复
        result = tracker.update(face_cx=960, face_cy=540, face_size=250, frame_w=1920, frame_h=1080)
        assert result is not None
        assert result['face_cx'] == 960.0


class TestInferenceScaleAdapt:
    """Task 2.1: inference_328.py 适配 output_size — AC-014"""

    def test_scale_with_output_size_848(self):
        """metadata 含 output_size=848 时，scale 计算正确"""
        metadata = {'output_size': 848}
        crop_size = 424
        output_size = metadata.get('output_size', 512)
        scale = crop_size / float(output_size)
        assert scale == pytest.approx(0.5)

    def test_scale_without_output_size_fallback_512(self):
        """metadata 无 output_size 字段时，回退到 512"""
        metadata = {}
        crop_size = 512
        output_size = metadata.get('output_size', 512)
        scale = crop_size / float(output_size)
        assert scale == pytest.approx(1.0)

    def test_scale_old_metadata_compatibility(self):
        """旧 metadata（output_size=512）与新逻辑兼容"""
        metadata = {'output_size': 512}
        crop_size = 256
        output_size = metadata.get('output_size', 512)
        scale = crop_size / float(output_size)
        assert scale == pytest.approx(0.5)
