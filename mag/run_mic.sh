for i in {000..029}; do
    python train.py -s /root/autodl-tmp/4DGS_MotionMag/data/synthetic/mic_5Hz/$i --port 6017 --expname synthetic/mic_5Hz/$i --configs /root/autodl-tmp/4DGS_MotionMag/arguments/dnerf/lego.py --start_checkpoint /root/autodl-tmp/4DGS_MotionMag/output/synthetic/mic_5Hz_000/chkpnt_fine_20000.pth --checkpoint_iterations 27000
done