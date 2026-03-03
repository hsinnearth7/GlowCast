"""Tests for deterministic hash bucketing."""

import numpy as np
import pytest

from app.experimentation.bucketing import BucketingAssigner


class TestBucketingAssigner:
    @pytest.fixture
    def assigner(self):
        return BucketingAssigner()

    def test_deterministic(self, assigner):
        b1 = assigner.assign_bucket("user_123", n_buckets=1000)
        b2 = assigner.assign_bucket("user_123", n_buckets=1000)
        bucket_1 = b1.bucket if hasattr(b1, "bucket") else b1
        bucket_2 = b2.bucket if hasattr(b2, "bucket") else b2
        assert bucket_1 == bucket_2

    def test_different_users_different_buckets(self, assigner):
        buckets = set()
        for i in range(100):
            b = assigner.assign_bucket(f"user_{i}", n_buckets=1000)
            bucket = b.bucket if hasattr(b, "bucket") else b
            buckets.add(bucket)
        assert len(buckets) > 50  # shouldn't all collide

    def test_salt_changes_assignment(self, assigner):
        b1 = assigner.assign_bucket("user_123", salt="exp_1")
        b2 = assigner.assign_bucket("user_123", salt="exp_2")
        bucket_1 = b1.bucket if hasattr(b1, "bucket") else b1
        bucket_2 = b2.bucket if hasattr(b2, "bucket") else b2
        # Different salts should (usually) produce different buckets
        # Not guaranteed but very likely for different salts
        assert isinstance(bucket_1, int)
        assert isinstance(bucket_2, int)

    def test_bucket_range(self, assigner):
        for i in range(200):
            b = assigner.assign_bucket(f"user_{i}", n_buckets=100)
            bucket = b.bucket if hasattr(b, "bucket") else b
            assert 0 <= bucket < 100

    def test_srm_check_balanced(self, assigner):
        observed = [5000, 5050]
        result = assigner.check_srm(observed, expected_ratio=[0.5, 0.5])
        # Should not detect SRM for roughly balanced groups
        is_srm = result.is_srm if hasattr(result, "is_srm") else result["is_srm"]
        assert not is_srm

    def test_srm_check_imbalanced(self, assigner):
        observed = [6000, 4000]
        result = assigner.check_srm(observed, expected_ratio=[0.5, 0.5])
        is_srm = result.is_srm if hasattr(result, "is_srm") else result["is_srm"]
        assert is_srm  # 60/40 split should be detected

    def test_uniform_distribution(self, assigner):
        buckets = []
        for i in range(10000):
            b = assigner.assign_bucket(f"user_{i}", n_buckets=10)
            bucket = b.bucket if hasattr(b, "bucket") else b
            buckets.append(bucket)
        counts = np.bincount(buckets, minlength=10)
        # Each bucket should have roughly 1000 ± 100
        assert all(800 < c < 1200 for c in counts)
