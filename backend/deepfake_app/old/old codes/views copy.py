from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, parser_classes
from django.core.files.storage import default_storage
from django.conf import settings
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import JsonResponse
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework.authtoken.models import Token
import os

from .models import VideoUpload, DetectionResult, DeepfakeImage, AwarenessContent, Report
from .serializers import VideoUploadSerializer, DetectionResultSerializer, AwarenessContentSerializer, ReportSerializer
from .deepfake_detection import generate_gradcam_and_ensemble_predict


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
        return Response({"error": "No file uploaded"}, status=400)

    file_type = uploaded_file.content_type

    if 'video' in file_type:
        video = VideoUpload(video=uploaded_file)
        video.save()
        return Response(
            {"message": "Video uploaded successfully", "video_id": video.id},
            status=201
        )

    elif 'image' in file_type:
        image_path = handle_uploaded_file(uploaded_file)

        # ✅ GET DICTIONARY RESULT
        result = generate_gradcam_and_ensemble_predict(
            request._request,
            image_path
        )

        # ✅ CHECK FOR ERRORS
        if "error" in result:
            return Response({
                "error": result["error"],
                "message": "Failed to analyze image"
            }, status=400)

        # ✅ EXTRACT VALUES FROM DICTIONARY
        prediction = result["label"]
        confidence = result["confidence"]
        grad_cam_path = result["gradcam_url"]
        winning_model = result["winning_model"]
        all_predictions = result["all_predictions"]

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
            "grad_cam_path": grad_cam_path,
            "winning_model": winning_model,
            "all_predictions": all_predictions
        }, status=201)

    else:
        return Response(
            {"error": "Unsupported file type. Only image and video files are allowed."},
            status=400
        )


# API to retrieve deepfake detection results for a file
@api_view(['GET'])
def get_results(request, file_id):
    """ API to retrieve deepfake detection results for a file """
    try:
        result = DeepfakeImage.objects.get(id=file_id)
        return Response({
            "prediction": result.prediction,
            "confidence": result.confidence,
            'grad_cam_path': result.grad_cam_path,
        })
    except DeepfakeImage.DoesNotExist:
        return Response({"error": "No results found for the given file ID"}, status=404)


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
        return Response({"error": "File ID is required"}, status=400)

    # Fetch the uploaded file based on the ID
    try:
        deepfake_image = DeepfakeImage.objects.get(id=file_id)
    except DeepfakeImage.DoesNotExist:
        return Response({"error": "File not found"}, status=404)

    # Perform analysis
    image_path = os.path.join(settings.MEDIA_ROOT, deepfake_image.image.name)
    
    # ✅ GET DICTIONARY RESULT (SAME AS upload_media)
    result = generate_gradcam_and_ensemble_predict(request, image_path)

    # ✅ CHECK FOR ERRORS
    if "error" in result:
        return Response({
            "error": result["error"],
            "message": "Failed to re-analyze image"
        }, status=400)

    # ✅ EXTRACT VALUES FROM DICTIONARY
    prediction = result["label"]
    confidence = result["confidence"]
    grad_cam_path = result["gradcam_url"]
    winning_model = result["winning_model"]
    all_predictions = result["all_predictions"]

    # Update the result in the database
    deepfake_image.prediction = prediction
    deepfake_image.confidence = confidence
    deepfake_image.save()

    return Response({
        "prediction": prediction,
        "confidence": confidence,
        'grad_cam_path': grad_cam_path,
        "winning_model": winning_model,
        "all_predictions": all_predictions,
        "message": "Analysis completed successfully."
    })


# ✅ 1. Login / Signup Page
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


# ✅ 1. User Dashboard - Fetch uploaded videos
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_dashboard(request):
    """ Fetch all videos uploaded by the authenticated user """
    user = request.user
    videos = VideoUpload.objects.filter(user=user)
    serializer = VideoUploadSerializer(videos, many=True)
    return Response(serializer.data)


# ✅ 2. Awareness Content - Fetch deepfake awareness articles
@api_view(['GET'])
def awareness_content(request):
    """ Fetch awareness articles on deepfake detection and misinformation """
    content = AwarenessContent.objects.all()
    serializer = AwarenessContentSerializer(content, many=True)
    return Response(serializer.data)


# ✅ 3. User Reports - Fetch past deepfake detection reports
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_reports(request):
    """ Fetch all deepfake detection reports for the authenticated user """
    user = request.user
    reports = Report.objects.filter(user=user)
    serializer = ReportSerializer(reports, many=True)
    return Response(serializer.data)

'''
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
#from .prnu_analysis import analyze_prnu
import os
from django.conf import settings

# ... existing imports and code ...

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def analyze_prnu_endpoint(request):
    """
    Standalone PRNU analysis endpoint
    
    POST /api/analyze-prnu/
    Body: { "file": <image file> }
    
    Returns: PRNU analysis results
    """
    uploaded_file = request.FILES.get('file')
    
    if not uploaded_file:
        return Response({"error": "No file uploaded"}, status=400)
    
    # Check if it's an image
    if 'image' not in uploaded_file.content_type:
        return Response({"error": "Only image files are supported"}, status=400)
    
    # Save temporarily
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'prnu_temp')
    os.makedirs(upload_dir, exist_ok=True)
    temp_path = os.path.join(upload_dir, uploaded_file.name)
    
    with open(temp_path, 'wb') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    
    # Run PRNU analysis
    try:
        result = analyze_prnu(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        if not result["success"]:
            return Response({
                "error": result.get("error", "PRNU analysis failed")
            }, status=400)
        
        return Response({
            "message": "PRNU analysis completed successfully",
            "prnu_analysis": result
        }, status=200)
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return Response({
            "error": f"PRNU analysis failed: {str(e)}"
        }, status=500)


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def full_analysis_with_prnu(request):
    """
    Combined endpoint: Main detection + PRNU analysis
    
    POST /api/full-analysis/
    Body: { "file": <image file> }
    
    Returns: Complete analysis including CNN ensemble + PRNU
    """
    uploaded_file = request.FILES.get('file')
    
    if not uploaded_file:
        return Response({"error": "No file uploaded"}, status=400)
    
    if 'image' not in uploaded_file.content_type:
        return Response({"error": "Only images supported"}, status=400)
    
    # Save file
    image_path = handle_uploaded_file(uploaded_file)
    
    # Run main detection pipeline
    main_result = generate_gradcam_and_ensemble_predict(
        request._request,
        image_path
    )
    
    if "error" in main_result:
        return Response({
            "error": main_result["error"]
        }, status=400)
    
    # Run PRNU analysis
    prnu_result = analyze_prnu(image_path)
    
    # Save to database
    deepfake_image = DeepfakeImage(
        image=uploaded_file,
        prediction=main_result["label"],
        confidence=main_result["confidence"]
    )
    deepfake_image.save()
    
    # Combine results
    return Response({
        "message": "Complete analysis finished",
        "main_detection": {
            "prediction": main_result["label"],
            "confidence": main_result["confidence"],
            "grad_cam_path": main_result["gradcam_url"],
            "winning_model": main_result["winning_model"],
            "decision_source": main_result["decision_source"],
            "forensic_scores": main_result["forensic_scores"],
            "cnn_scores": main_result["cnn_scores"]
        },
        "prnu_analysis": prnu_result if prnu_result["success"] else {"error": "PRNU failed"},
        "combined_assessment": generate_combined_verdict(main_result, prnu_result)
    }, status=200)


def generate_combined_verdict(main_result, prnu_result):
    """Combine main detection and PRNU for final verdict"""
    
    main_label = main_result["label"]
    main_conf = main_result["confidence"]
    
    if not prnu_result["success"]:
        return {
            "verdict": main_label,
            "confidence": main_conf,
            "note": "PRNU analysis unavailable, relying on CNN ensemble only"
        }
    
    prnu_score = prnu_result["normalized_scores"]["overall_prnu_score"]
    
    # Agreement check
    prnu_suggests_fake = prnu_score >= 0.50
    main_suggests_fake = main_label == "Fake"
    
    if prnu_suggests_fake == main_suggests_fake:
        # Both agree
        combined_confidence = min(0.95, (main_conf + (1 - abs(prnu_score - 0.5) * 2)) / 2)
        return {
            "verdict": main_label,
            "confidence": round(combined_confidence, 4),
            "agreement": "Strong - both CNN and PRNU agree",
            "note": "High confidence due to corroborating evidence"
        }
    else:
        # Disagree
        return {
            "verdict": main_label,
            "confidence": round(main_conf * 0.8, 4),  # Reduce confidence
            "agreement": "Weak - CNN and PRNU disagree",
            "note": f"CNN says {main_label}, but PRNU suggests {'Fake' if prnu_suggests_fake else 'Real'}. Further review recommended."
        }

'''