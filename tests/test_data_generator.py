"""Tests for the GlowCast data generator engine."""

import pandas as pd

from app.data.data_generator import GlowCastDataGenerator


class TestGeneratorInit:
    def test_default_init(self):
        gen = GlowCastDataGenerator(n_skus=10, n_days=30)
        assert gen.n_skus == 10
        assert gen.n_days == 30
        assert gen.start_date == pd.Timestamp("2022-01-01")

    def test_custom_dates(self):
        gen = GlowCastDataGenerator(n_skus=10, n_days=30, start_date="2023-06-01")
        assert gen.start_date == pd.Timestamp("2023-06-01")
        assert len(gen.dates) == 30


class TestDimProduct:
    def test_generates_correct_count(self, small_generator):
        count = len(small_generator.dim_product)
        assert count >= 50 - 5 and count <= 50 + 5

    def test_sku_ids_unique(self, small_generator):
        assert small_generator.dim_product["sku_id"].is_unique

    def test_sku_id_format(self, small_generator):
        assert all(s.startswith("SKU_") for s in small_generator.dim_product["sku_id"])

    def test_all_concerns_present(self, small_generator):
        from app.data.segment_genes import CONCERNS
        present = set(small_generator.dim_product["concern"].unique())
        assert present == set(CONCERNS)

    def test_price_positive(self, small_generator):
        assert (small_generator.dim_product["retail_price"] > 0).all()
        assert (small_generator.dim_product["unit_cost"] > 0).all()

    def test_margin_valid(self, small_generator):
        margins = small_generator.dim_product["gross_margin"]
        assert (margins > 0).all()
        assert (margins < 1).all()


class TestDimLocation:
    def test_12_fcs(self, small_generator):
        assert len(small_generator.dim_location) == 12

    def test_5_countries(self, small_generator):
        countries = small_generator.dim_location["country"].unique()
        assert len(countries) == 5

    def test_lat_lon_valid(self, small_generator):
        assert (small_generator.dim_location["lat"].between(-90, 90)).all()
        assert (small_generator.dim_location["lon"].between(-180, 180)).all()


class TestDimWeather:
    def test_weather_generated(self, small_generator):
        assert small_generator.dim_weather is not None
        assert len(small_generator.dim_weather) > 0

    def test_12_regions(self, small_generator):
        regions = small_generator.dim_weather["region"].nunique()
        assert regions == 12

    def test_humidity_clipped(self, small_generator):
        assert (small_generator.dim_weather["humidity_pct"] >= 10).all()
        assert (small_generator.dim_weather["humidity_pct"] <= 100).all()

    def test_seasons_present(self, small_generator):
        seasons = set(small_generator.dim_weather["season"].unique())
        # 90 days might not cover all seasons, but should have at least 1
        assert len(seasons) >= 1


class TestFactSales:
    def test_sales_generated(self, small_generator):
        assert small_generator.fact_sales is not None
        assert len(small_generator.fact_sales) > 0

    def test_order_id_format(self, small_generator):
        assert all(s.startswith("ORD_") for s in small_generator.fact_sales["order_id"])

    def test_units_non_negative(self, small_generator):
        assert (small_generator.fact_sales["units_sold"] >= 0).all()

    def test_revenue_non_negative(self, small_generator):
        assert (small_generator.fact_sales["revenue"] >= 0).all()

    def test_discount_rates_valid(self, small_generator):
        rates = small_generator.fact_sales["discount_rate"].unique()
        for r in rates:
            assert 0 < r <= 1.0

    def test_is_return_binary(self, small_generator):
        assert set(small_generator.fact_sales["is_return"].unique()).issubset({0, 1})


class TestFactSocial:
    def test_social_generated(self, small_generator):
        assert small_generator.fact_social is not None
        assert len(small_generator.fact_social) > 0

    def test_viral_rate_around_3pct(self, small_generator):
        viral_pct = small_generator.fact_social["is_viral"].mean()
        assert 0.01 < viral_pct < 0.08  # loose bounds for small sample

    def test_sentiment_range(self, small_generator):
        assert (small_generator.fact_social["sentiment_score"] >= -1).all()
        assert (small_generator.fact_social["sentiment_score"] <= 1).all()


class TestFactInventory:
    def test_inventory_generated(self, small_generator):
        assert small_generator.fact_inventory is not None
        assert len(small_generator.fact_inventory) > 0

    def test_batch_id_format(self, small_generator):
        assert all(s.startswith("B-") for s in small_generator.fact_inventory["batch_id"])

    def test_expiry_after_manufacturing(self, small_generator):
        inv = small_generator.fact_inventory
        assert (inv["expiry_date"] > inv["manufacturing_date"]).all()

    def test_units_positive(self, small_generator):
        assert (small_generator.fact_inventory["units_on_hand"] > 0).all()


class TestFactFulfillment:
    def test_fulfillment_generated(self, small_generator):
        assert small_generator.fact_fulfillment is not None
        assert len(small_generator.fact_fulfillment) > 0

    def test_fulfillment_types(self, small_generator):
        types = set(small_generator.fact_fulfillment["fulfillment_type"].unique())
        assert types.issubset({"Local_Fulfillment", "Cross_Zone_Fulfillment"})

    def test_cross_zone_minority(self, small_generator):
        cz = (small_generator.fact_fulfillment["fulfillment_type"] == "Cross_Zone_Fulfillment").mean()
        assert cz < 0.30  # should be around 15%


class TestFactReviews:
    def test_reviews_generated(self, small_generator):
        assert small_generator.fact_reviews is not None
        assert len(small_generator.fact_reviews) > 0

    def test_star_rating_range(self, small_generator):
        assert (small_generator.fact_reviews["star_rating"] >= 1).all()
        assert (small_generator.fact_reviews["star_rating"] <= 5).all()

    def test_aspect_sentiment_range(self, small_generator):
        assert (small_generator.fact_reviews["aspect_sentiment"] >= -1).all()
        assert (small_generator.fact_reviews["aspect_sentiment"] <= 1).all()


class TestNixtlaFormat:
    def test_to_nixtla_format(self, small_generator):
        Y_df, S_df = small_generator.to_nixtla_format()
        assert "unique_id" in Y_df.columns
        assert "ds" in Y_df.columns
        assert "y" in Y_df.columns
        assert (Y_df["y"] >= 0).all()

    def test_hierarchy_columns(self, small_generator):
        _, S_df = small_generator.to_nixtla_format()
        assert "sku_id" in S_df.columns
        assert "fc_id" in S_df.columns
        assert "country" in S_df.columns
        assert "national" in S_df.columns


class TestReproducibility:
    def test_deterministic_hash(self):
        gen1 = GlowCastDataGenerator(n_skus=10, n_days=30, seed=42)
        gen1.generate_all()
        hash1 = gen1.compute_data_hash()

        gen2 = GlowCastDataGenerator(n_skus=10, n_days=30, seed=42)
        gen2.generate_all()
        hash2 = gen2.compute_data_hash()

        assert hash1 == hash2

    def test_different_seed_different_hash(self):
        gen1 = GlowCastDataGenerator(n_skus=10, n_days=30, seed=42)
        gen1.generate_all()

        gen2 = GlowCastDataGenerator(n_skus=10, n_days=30, seed=99)
        gen2.generate_all()

        assert gen1.compute_data_hash() != gen2.compute_data_hash()


class TestSummary:
    def test_summary_all_tables(self, small_generator):
        summary = small_generator.summary()
        assert len(summary) == 9
        for name, count in summary.items():
            assert count > 0, f"{name} has 0 rows"
