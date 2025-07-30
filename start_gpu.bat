@echo off
echo Starting FAISS GPU Product Search System...
set USE_FAISS_GPU=true
set CUDA_VISIBLE_DEVICES=0
python app.py