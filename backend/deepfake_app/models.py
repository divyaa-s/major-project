from django.db import models
from django.contrib.auth.models import User

# Video Upload Model
class VideoUpload(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)  # Allow null values temporarily
    video = models.FileField(upload_to="uploads/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

# Deepfake Detection Result Model
class DetectionResult(models.Model):
    video = models.ForeignKey(VideoUpload, on_delete=models.CASCADE)
    is_fake = models.BooleanField()
    confidence_score = models.FloatField()
    heatmap_image = models.ImageField(upload_to="heatmaps/", null=True, blank=True)

# Awareness Content Model
class AwarenessContent(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

# History & Report Storage
class Report(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video = models.ForeignKey(VideoUpload, on_delete=models.CASCADE)
    report_file = models.FileField(upload_to="reports/")
    created_at = models.DateTimeField(auto_now_add=True)

from django.db import models


class DeepfakeImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    prediction = models.CharField(max_length=10)
    confidence = models.FloatField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    # Explainability & visuals
    gradcam_url = models.URLField(blank=True, null=True)
    radar_url = models.URLField(blank=True, null=True)
    bar_plot_url = models.URLField(blank=True, null=True)

    # Decision trace
    decision_source = models.CharField(max_length=128, blank=True, null=True)

    # Watermark forensic signal
    watermark_prob = models.FloatField(blank=True, null=True)
    watermark_detected = models.BooleanField(default=False)

    def __str__(self):
        return f"DeepfakeImage {self.id} - {self.prediction} ({self.confidence:.2f})"
