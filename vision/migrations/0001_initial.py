from __future__ import annotations

from decimal import Decimal

from django.conf import settings
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="Food",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("name", models.CharField(max_length=255, unique=True)),
                ("category", models.CharField(blank=True, max_length=128)),
                ("nutritional_info", models.JSONField(blank=True, default=dict)),
                ("is_active", models.BooleanField(default=True)),
                ("default_safety_note", models.TextField(blank=True)),
            ],
            options={
                "ordering": ["name"],
            },
        ),
        migrations.CreateModel(
            name="PregnancyStage",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("name", models.CharField(max_length=128)),
                ("start_week", models.PositiveIntegerField()),
                ("end_week", models.PositiveIntegerField()),
                ("description", models.TextField(blank=True)),
            ],
            options={
                "ordering": ["start_week"],
                "unique_together": {("start_week", "end_week")},
            },
        ),
        migrations.CreateModel(
            name="ResponseStyle",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("name", models.CharField(max_length=64, unique=True)),
                ("prompt", models.TextField()),
                ("description", models.CharField(blank=True, max_length=255)),
                ("is_default", models.BooleanField(default=False)),
            ],
            options={
                "ordering": ["name"],
            },
        ),
        migrations.CreateModel(
            name="UserPregnancyProfile",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("current_week", models.PositiveIntegerField(default=1)),
                ("pre_pregnancy_bmi", models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ("weight_gain_kg", models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ("stage_cached_id", models.BigIntegerField(blank=True, null=True)),
                ("stage_cached_label", models.CharField(blank=True, max_length=255)),
                ("stage_cached_until", models.DateTimeField(blank=True, null=True)),
                ("user", models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name="pregnancy_profile", to=settings.AUTH_USER_MODEL)),
            ],
            options={
                "ordering": ["-updated_at"],
            },
        ),
        migrations.CreateModel(
            name="NutrientRequirement",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("nutrient_name", models.CharField(max_length=128)),
                ("daily_value", models.DecimalField(decimal_places=2, max_digits=10)),
                ("unit", models.CharField(default="mg", max_length=32)),
                ("description", models.TextField(blank=True)),
                ("stage", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="nutrient_requirements", to="vision.pregnancystage")),
            ],
            options={
                "ordering": ["nutrient_name"],
                "unique_together": {("stage", "nutrient_name")},
            },
        ),
        migrations.CreateModel(
            name="FoodLog",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("meal_type", models.CharField(choices=[("breakfast", "아침"), ("lunch", "점심"), ("dinner", "저녁"), ("snack", "간식")], max_length=16)),
                ("portion", models.DecimalField(decimal_places=2, default=1, max_digits=6)),
                ("logged_at", models.DateTimeField(db_index=True, default=django.utils.timezone.now)),
                ("notes", models.TextField(blank=True)),
                ("food", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="food_logs", to="vision.food")),
                ("user", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="food_logs", to=settings.AUTH_USER_MODEL)),
            ],
            options={
                "ordering": ["-logged_at"],
            },
        ),
        migrations.CreateModel(
            name="FoodRecommendation",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("priority", models.PositiveSmallIntegerField(validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(10)])),
                ("is_safe", models.BooleanField(default=True)),
                ("safety_info", models.TextField(blank=True)),
                ("nutritional_advice", models.TextField(blank=True)),
                ("reasoning", models.TextField(blank=True)),
                ("food", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="recommendations", to="vision.food")),
                ("user", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="food_recommendations", to=settings.AUTH_USER_MODEL)),
            ],
            options={
                "ordering": ["priority", "-created_at"],
                "unique_together": {("user", "food")},
            },
        ),
        migrations.CreateModel(
            name="FoodRating",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("rating", models.PositiveSmallIntegerField(validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(5)])),
                ("comment", models.TextField(blank=True)),
                ("food", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="ratings", to="vision.food")),
                ("user", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="food_ratings", to=settings.AUTH_USER_MODEL)),
            ],
            options={
                "ordering": ["-created_at"],
                "unique_together": {("user", "food")},
            },
        ),
        migrations.CreateModel(
            name="FoodRecognitionLog",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("image_sha", models.CharField(db_index=True, max_length=128)),
                ("food_name", models.CharField(blank=True, max_length=255)),
                ("confidence_score", models.DecimalField(decimal_places=3, default=Decimal("0.0"), max_digits=4, validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(1)])),
                ("raw_response", models.JSONField(blank=True, default=dict)),
                ("status", models.CharField(choices=[("success", "성공"), ("failure", "실패")], default="success", max_length=16)),
                ("error_message", models.TextField(blank=True)),
                ("image_placeholder", models.CharField(blank=True, max_length=255)),
                ("user", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="vision_logs", to=settings.AUTH_USER_MODEL)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.AddIndex(
            model_name="food",
            index=models.Index(fields=["name"], name="vision_food_name_idx"),
        ),
        migrations.AddIndex(
            model_name="foodlog",
            index=models.Index(fields=["user", "logged_at"], name="vision_foodlog_user_logged_idx"),
        ),
        migrations.AddIndex(
            model_name="foodrecognitionlog",
            index=models.Index(fields=["user", "created_at"], name="vision_recog_user_created_idx"),
        ),
        migrations.AddIndex(
            model_name="foodrecognitionlog",
            index=models.Index(fields=["image_sha"], name="vision_recog_image_sha_idx"),
        ),
        migrations.AddIndex(
            model_name="pregnancystage",
            index=models.Index(fields=["start_week", "end_week"], name="vision_stage_week_idx"),
        ),
    ]


