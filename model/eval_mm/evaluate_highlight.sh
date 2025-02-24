echo "start evaluation..."
NVIDIA_VISIBLE_DEVICES="0,1" python3 inf.py
echo "evaluate finished."
NVIDIA_VISIBLE_DEVICES="0,1" python3 highlight.py
echo "highlight finished."