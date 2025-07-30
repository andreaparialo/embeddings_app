@echo off
echo Starting OpenCLIP GPU Product Search System...
set USE_FAISS_GPU=true
set CUDA_VISIBLE_DEVICES=0
python app_openclip.py