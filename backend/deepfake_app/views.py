from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, parser_classes
from django.core.files.storage import default_storage
from django.conf import settings
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework.authtoken.models import Token
import os
import logging

from .models import VideoUpload, DetectionResult, DeepfakeImage, AwarenessContent, Report
from .serializers import VideoUploadSerializer, DetectionResultSerializer, AwarenessContentSerializer, ReportSerializer
from .deepfake_detection import generate_gradcam_and_ensemble_predict

logger = logging.getLogger(__name__)

# ✅ HELPER FUNCTION TO SAVE UPLOADED FILES
def handle_uploaded_file(file):
    """Save uploaded file and return its path"""
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    img_path = os.path.join(upload_dir, file.name)
    with open(img_path, 'wb') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return img_path


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_media(request):
    uploaded_file = request.FILES.get('file')

    if not uploaded_file:
        return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

    file_type = uploaded_file.content_type or ""

    if 'video' in file_type.lower():
        video = VideoUpload(video=uploaded_file)
        video.save()
        return Response(
            {"message": "Video uploaded successfully", "video_id": video.id},
            status=status.HTTP_201_CREATED
        )

    elif 'image' in file_type.lower():
        image_path = handle_uploaded_file(uploaded_file)

        try:
            # ✅ GET DICTIONARY RESULT — now with true_label
            result = generate_gradcam_and_ensemble_predict(
                request,                    # ← fixed: use request (not _request)
                image_path,
                true_label="Unknown"        # ← added: required by updated deepfake function
            )

            # ✅ CHECK FOR ERRORS
            if "error" in result:
                return Response({
                    "error": result["error"],
                    "message": "Failed to analyze image"
                }, status=status.HTTP_400_BAD_REQUEST)

            # ✅ EXTRACT VALUES FROM DICTIONARY
            prediction = result.get("label")
            confidence = result.get("confidence")

            grad_cam_path = result.get("gradcam_url")
            radar_url = result.get("radar_url")
            bar_plot_url = result.get("plot_url")

            winning_model = result.get("winning_model")
            all_predictions = result.get("all_predictions")
            decision_source = result.get("decision_source")
            watermark = result.get("watermark")


            # Save to database
            deepfake_image = DeepfakeImage(
                image=uploaded_file,
                prediction=prediction,
                confidence=confidence
            )
            deepfake_image.save()

            # ✅ RETURN COMPREHENSIVE RESPONSE
            return Response({
                "message": "Image uploaded and analyzed successfully",

                "prediction": prediction,
                "confidence": confidence,
                "decision_source": decision_source,

                # 🔥 ALL VISUALS
                "gradcam_url": grad_cam_path,
                "radar_url": radar_url,
                "bar_plot_url": bar_plot_url,
                "watermark": watermark,   # ← 🔥 THIS IS THE MISSING PIECE

                "winning_model": winning_model,
                "all_predictions": all_predictions,

                "image_id": deepfake_image.id
            }, status=status.HTTP_201_CREATED)



        except Exception as e:
            logger.exception("Error during image analysis")
            return Response({
                "error": "Analysis failed",
                "detail": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    else:
        return Response(
            {"error": "Unsupported file type. Only image and video files are allowed."},
            status=status.HTTP_400_BAD_REQUEST
        )


# API to retrieve deepfake detection results for a file
@api_view(['GET'])
def get_results(request, file_id):
    """Retrieve full deepfake detection results for a given image"""
    try:
        result = DeepfakeImage.objects.get(id=file_id)

        return Response({
            "prediction": result.prediction,
            "confidence": result.confidence,

            # Visual explanations
            "gradcam_url": getattr(result, "gradcam_url", None),
            "radar_url": getattr(result, "radar_url", None),
            "bar_plot_url": getattr(result, "bar_plot_url", None),

            # Decision explainability
            "decision_source": getattr(result, "decision_source", None),

            # Watermark forensic signal
            "watermark": {
                "probability": getattr(result, "watermark_prob", None),
                "detected": getattr(result, "watermark_detected", None)
            }
        }, status=200)

    except DeepfakeImage.DoesNotExist:
        return Response(
            {"error": "No results found for the given file ID"},
            status=404
        )



# ✅ FIXED: API endpoint to re-analyze uploaded file
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def analyze_uploaded_file(request):
    """
    API to analyze the uploaded file and return deepfake prediction and confidence.
    This can be used to re-trigger analysis after file upload.
    """
    file_id = request.data.get("file_id")
    if not file_id:
        return Response({"error": "File ID is required"}, status=status.HTTP_400_BAD_REQUEST)

    # Fetch the uploaded file based on the ID
    try:
        deepfake_image = DeepfakeImage.objects.get(id=file_id)
    except DeepfakeImage.DoesNotExist:
        return Response({"error": "File not found"}, status=404)

    try:
        # Perform analysis
        image_path = os.path.join(settings.MEDIA_ROOT, deepfake_image.image.name)
        
        result = generate_gradcam_and_ensemble_predict(
            request,
            image_path,
            true_label="Unknown"           # ← added
        )

        if "error" in result:
            return Response({
                "error": result["error"],
                "message": "Failed to re-analyze image"
            }, status=status.HTTP_400_BAD_REQUEST)

        prediction = result.get("label")
        confidence = result.get("confidence")
        grad_cam_path = result.get("gradcam_url")
        winning_model = result.get("winning_model")
        all_predictions = result.get("all_predictions")
        grad_cam_path = result.get("gradcam_url")
        radar_url = result.get("radar_url")
        bar_plot_url = result.get("plot_url")
        decision_source = result.get("decision_source")
        watermark = result.get("watermark")

        # Update the result in the database
        deepfake_image.prediction = prediction
        deepfake_image.confidence = confidence
        deepfake_image.save()

        return Response({
            "prediction": prediction,
            "confidence": confidence,
            "decision_source": decision_source,

            "gradcam_url": grad_cam_path,
            "radar_url": radar_url,
            "bar_plot_url": bar_plot_url,
            "watermark": watermark,
            "winning_model": winning_model,
            "all_predictions": all_predictions,

            "message": "Analysis completed successfully."
        }, status=status.HTTP_200_OK)


    except Exception as e:
        logger.exception("Re-analysis failed")
        return Response({
            "error": "Re-analysis failed",
            "detail": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ────────────────────────────────────────────────
# Auth endpoints (unchanged)
# ────────────────────────────────────────────────
@api_view(['POST'])
def signup(request):
    username = request.data.get('username')
    email = request.data.get('email')
    password = request.data.get('password')

    if User.objects.filter(username=username).exists():
        return Response({"error": "Username already taken"}, status=400)

    user = User.objects.create_user(username=username, email=email, password=password)
    token, _ = Token.objects.get_or_create(user=user)
    return Response({"token": token.key, "message": "Signup successful"})


@api_view(['POST'])
def login(request):
    username = request.data.get('username')
    password = request.data.get('password')
    user = authenticate(username=username, password=password)

    if user:
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"token": token.key, "message": "Login successful"})
    return Response({"error": "Invalid credentials"}, status=400)


# ────────────────────────────────────────────────
# Dashboard & content endpoints (unchanged)
# ────────────────────────────────────────────────
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_dashboard(request):
    """ Fetch all videos uploaded by the authenticated user """
    user = request.user
    videos = VideoUpload.objects.filter(user=user)
    serializer = VideoUploadSerializer(videos, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def awareness_content(request):
    """ Fetch awareness articles on deepfake detection and misinformation """
    content = AwarenessContent.objects.all()
    serializer = AwarenessContentSerializer(content, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_reports(request):
    """Fetch all deepfake detection reports for the authenticated user"""
    user = request.user
    reports = Report.objects.filter(user=user)

    response = []
    for r in reports:
        response.append({
            "id": r.id,
            "file_name": r.file_name,
            "analysis_date": r.created_at,

            "prediction": r.prediction,
            "confidence": r.confidence,
            "decision_source": r.decision_source,

            # Visual evidence
            "gradcam_url": r.gradcam_url,
            "radar_url": r.radar_url,
            "bar_plot_url": r.bar_plot_url,

            # Watermark forensic signal
            "watermark": {
                "probability": r.watermark_prob,
                "detected": r.watermark_detected
            }
        })

    return Response(response, status=200)
