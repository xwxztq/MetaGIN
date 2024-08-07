# MetaGIN: A Lightweight Framework for Molecular Property Prediction



Recent advancements in AI-based synthesis of small molecules have led to the creation of extensive databases, housing billions of small molecules. Given this vast scale, traditional quantum chemistry (QC) methods become inefficient for determining the chemical and physical properties of such an extensive array of molecules. As these properties are key to new drug research and development, deep learning techniques have recently been developed. Here, we present MetaGIN, a lightweight framework designed for molecular property prediction.

While traditional GNN models with 1-hop edges (i.e., covalent bonds) are sufficient for abstract graph representation, they are inadequate for capturing 3D features. Our MetaGIN model shows that including 2-hop and 3-hop edges (representing bond and torsion angles, respectively) is crucial to fully comprehend the intricacies of 3D molecules. Moreover, MetaGIN is a streamlined model with fewer than 10 million parameters, making it ideal for fine-tuning on a single GPU. It also adopts the widely acknowledged MetaFormer framework, which has consistently shown high accuracy in many computer vision tasks.

Through our experiments, MetaGIN achieves a mean absolute error (MAE) of 0.0851 with just 8.87M parameters on the PCQM4Mv2 dataset. Furthermore, MetaGIN outperforms leading techniques, showcasing superior performance across several datasets in the MoleculeNet benchmark.


## Usage

### Installation

To install MetaGIN, clone the repository and install the required packages:

```bash
conda create -n metagin python=3.10
conda activate metagin
conda install -y pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y pyg pytorch-scatter pytorch-sparse -c pyg
conda install -y ogb rdkit h5py
```


### Download the checkpoint

```bash
wget https://zenodo.org/records/13147277/files/metagin_wide.pt
```

### Using the model

```bash
python src/predict.py --model wide31 --checkpoint metagin_wide.pt --smiles_file smiles.txt
```
