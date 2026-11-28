

def test_other_person_overlap_marks_blocked_view():
    import numpy as np

    from boulder.pose.rtmpose_backend import other_person_overlaps

    climber = np.array([[100, 100], [200, 400]], dtype=float)      # box 100x300
    passerby = np.array([[150, 250], [400, 600]], dtype=float)     # covers the lower half
    far_away = np.array([[800, 100], [900, 400]], dtype=float)
    ok = np.array([0.9, 0.9])
    assert other_person_overlaps(np.stack([climber, passerby]), np.stack([ok, ok]), 0)
    assert not other_person_overlaps(np.stack([climber, far_away]), np.stack([ok, ok]), 0)
    assert not other_person_overlaps(np.stack([climber]), np.stack([ok]), 0)
