
### Environment
```
conda create -n deca
pip install -r requirements.txt
```

### Downloads DECA model
```
./fetch_data.sh
```

### Run
Run the following cmd to generate a custom facial point cloud.
```
cd FantasyID
python -m  decalib.process_pcd --input_path ./assets/bob.jpg --output_path ./assets/bob.ply
```

