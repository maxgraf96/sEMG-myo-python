# Multimodal Hand Tracking for Extended Reality Musical Instruments (Deep Learning Model)

This repository contains the machine learning code of the multimodal XR Hand Tracking system for Extended Reality Musical Instruments (XRMIs), as described in our research paper "Combining Vision and EMG-Based Hand Tracking for Extended Reality Musical Instruments".
Included are files for processing and analyzing sEMG (surface electromyography) data using the Myo armband with Python.
The code is part of a research project at Queen Mary University of London, which aims to develop a multimodal hand tracking system for XRMIs.

Note: The Unity implementation for the multimodal hand tracking system can be found [here](https://github.com/maxgraf96/sEMG-myo-unity).

## Project Structure

- `data/`: Training data from several recording sessions.
- `main_rnn.py`: The main training loop for the RNN model.
- `exporter_pytorch_to_onnx.py`: Script to export the trained PyTorch model to ONNX format.
- `requirements.txt`: List of dependencies.
- `DataModule.py`: Data handling module for the project.
- `hyperparameters.py`: Hyperparameters configuration.
- `README.md`: Documentation and setup instructions (this file).

## Setup Instructions

### Prerequisites

- Git
- Python (version >= 3.8)

### Installation Steps

1. Clone the repository:
`git clone https://github.com/maxgraf96/sEMG-myo-python`

2. Navigate to the cloned directory:
`cd sEMG-myo-python`

3. Ensure that you have the correct version of Python installed (>= 3.8).

4. Install the required dependencies:
`pip install -r requirements.txt`


### Usage

To run the main training loop, execute the following command:
`python main_rnn.py`

After training the model, if you wish to export it to ONNX format, run:
`python exporter_pytorch_to_onnx.py`

## Contributing

Feel free to fork the repository, make changes, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Contact

- Max Graf - [max.graf@qmul.ac.uk](mailto:max.graf@qmul.ac.uk)
- Project Link: [https://github.com/maxgraf96/sEMG-myo-python](https://github.com/maxgraf96/sEMG-myo-python)

## Citation
If you use this work, please cite
   ```
   @misc{graf2023combining,
      title={Combining Vision and EMG-Based Hand Tracking for Extended Reality Musical Instruments}, 
      author={Max Graf and Mathieu Barthet},
      year={2023},
      eprint={2307.10203},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
  }
   ```
