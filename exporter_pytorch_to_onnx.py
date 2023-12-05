import time
import torch
import onnx

from hyperparameters import DATA_LEN, LOSS_INDICES, SEQ_LEN, WAVELET_FEATURE_LEN
from sEMGRNN import SEMGRNN, model_name

device = torch.device("cpu")

if __name__ == '__main__':

    with torch.no_grad():
        # Load model
        model = SEMGRNN(onnx_export=True).to(device)
        model.load_state_dict(torch.load(model_name + ".pt", map_location="cpu"))
        model.eval()

        # Define sample input
        input_sample = torch.rand(
        1,               # Batch size 
        6                # 6 time/freq domain features 
        + WAVELET_FEATURE_LEN             # wavelet features
        + DATA_LEN       # DATA_LEN EMG samples
        + DATA_LEN,      # DATA_LEN wrist angle samples
        8).to(device)    
        # input_sample = torch.randn((1, DATA_LEN, 8)).to(device)
        # input_sample = torch.randn((1, 1, SEQ_LEN - 1, 8)).to(device)
        
        # Take time
        current_time = time.time()

        # Export to ONNX
        print("Exporting to ONNX...")
        filepath = model_name + ".onnx"
        
        model.to_onnx(
            filepath,
            input_sample,
            export_params=True, 
            opset_version=17, 
            input_names=["input"], 
            output_names=["output"],
            verbose=True)

        # Print time difference in seconds
        print("Time taken:", time.time() - current_time)

        print("Success! Loading ONNX model...")
        # Load with onnx
        onnx_model = onnx.load(model_name + ".onnx")
        onnx.checker.check_model(onnx_model)
        print("ONNX model loaded successfully!")
        print(onnx.helper.printable_graph(onnx_model.graph))