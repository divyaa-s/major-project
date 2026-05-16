from rest_framework import serializers
from .models import VideoUpload, DetectionResult, AwarenessContent, Report

# Serializer for Video Upload
class VideoUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoUpload
        fields = '__all__'

# Serializer for Deepfake Detection Results
class DetectionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectionResult
        fields = '__all__'

# ✅ Serializer for Awareness Content (Ensure this exists!)
class AwarenessContentSerializer(serializers.ModelSerializer):
    class Meta:
        model = AwarenessContent
        fields = '__all__'

# Serializer for User Reports
class ReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Report
        fields = '__all__'
