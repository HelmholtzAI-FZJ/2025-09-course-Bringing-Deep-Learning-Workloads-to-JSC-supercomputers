---
author: Alexandre Strube // Sabrina Benassou 
title: Bringing Deep Learning Workloads to JSC supercomputers
subtitle: Data loading
date: June 25th, 2025
---

### Schedule for day 2

| Time          | Title                |
| ------------- | -----------          |
| 13:00 - 13:10 | Welcome, questions   |
| 13:10 - 14:10 | Data loading |
| 14:10 - 14:25 | Coffee Break (flexible) |
| 14:25 - 17:00 | Parallelize Training |

---

## Let's talk about DATA

![](images/data.jpeg)

--- 

## I/O is separate and shared

- All compute nodes of all supercomputers see the same files
- Performance tradeoff between shared acessibility and speed
- Our I/O server is almost a supercomputer by itself
    ![JSC Supercomputer Stragegy](images/machines.png){height=350pt}

---

## Where do I keep my files?

- Always store your code in the project1 folder (**`$PROJECT_projectname`** ). In our case 

    ```bash
    /p/project1/training2543/$USER
    ```

- Store data in the scratch directory for faster I/O access (**`$SCRATCH_projectname`**). ⚠️**Files in scratch are deleted after 90 days of inactivity.**
    
    ```bash
    /p/scratch/training2543/$USER
    ```

- Store the data in [`$DATA_dataset`](https://judoor.fz-juelich.de/projects/datasets/) for a more permanent location. 

    ```bash
    /p/data1/datasets
    ```

---

## Data loading

- We have CPUs and lots of memory - let's use them
- If your dataset is relatively small (< 500 GB) and can fit into the working memory (RAM) of each compute node (along with the program state), you can store it in **``/dev/shm``**. This is a special filesystem that uses RAM for storage, making it extremely fast for data access. ⚡️
- For bigger datasets (> 500 GB) you have many strategies:
    - Hierarchical Data Format 5 (HDF5)
    - Apache Arrow
    - NVIDIA Data Loading Library (DALI)
    - SquashFS


---

## Inodes 
- Inodes (Index Nodes) are data structures that store metadata about files and directories.
- Unique identification of files and directories within the file system.
- Efficient management and retrieval of file metadata.
- Essential for file operations like opening, reading, and writing.
- **Limitations**:
  - **Fixed Number**: Limited number of inodes; no new files if exhausted, even with free disk space.
  - **Space Consumption**: Inodes consume disk space, balancing is needed for efficiency.
![](images/inodes.png)

---

## Data loading

- In this course, we provide you with some examples on how to create and HDF5 and pyarrow files.

- We need to download some code:

    ```bash
    cd $HOME/course
    git clone https://github.com/HelmholtzAI-FZJ/2025-06-course-Bringing-Deep-Learning-Workloads-to-JSC-supercomputers.git
    ```

- Move to the correct folder:

    ```
    cd 2025-06-course-Bringing-Deep-Learning-Workloads-to-JSC-supercomputers/code/dataloading/
    ```

- We used the ImageNet dataset for the examples.

---

## The ImageNet dataset
#### Large Scale Visual Recognition Challenge (ILSVRC)
- An image dataset organized according to the [WordNet hierarchy](https://wordnet.princeton.edu). 
- Extensively used in algorithms for object detection and image classification at large scale. 
- It has 1000 classes, that comprises 1.2 million images for training, and 50,000 images for the validation set.

![](images/imagenet_banner.jpeg)

---

## The ImageNet dataset

```bash
ILSVRC
|-- Data/
    `-- CLS-LOC
        |-- test
        |-- train
        |   |-- n01440764
        |   |   |-- n01440764_10026.JPEG
        |   |   |-- n01440764_10027.JPEG
        |   |   |-- n01440764_10029.JPEG
        |   |-- n01695060
        |   |   |-- n01695060_10009.JPEG
        |   |   |-- n01695060_10022.JPEG
        |   |   |-- n01695060_10028.JPEG
        |   |   |-- ...
        |   |...
        |-- val
            |-- ILSVRC2012_val_00000001.JPEG  
            |-- ILSVRC2012_val_00016668.JPEG  
            |-- ILSVRC2012_val_00033335.JPEG      
            |-- ...
```
---

## The ImageNet dataset
imagenet_train.pkl

```bash 
{
    'ILSVRC/Data/CLS-LOC/train/n03146219/n03146219_8050.JPEG': 524,
    'ILSVRC/Data/CLS-LOC/train/n03146219/n03146219_12728.JPEG': 524,
    'ILSVRC/Data/CLS-LOC/train/n03146219/n03146219_9736.JPEG': 524,
    ...
    'ILSVRC/Data/CLS-LOC/train/n03146219/n03146219_7460.JPEG': 524,
    ...
 }
```

imagenet_val.pkl

```bash
{
    'ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00008838.JPEG': 785,
    'ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00008555.JPEG': 129,
    'ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00028410.JPEG': 968,
    ...
    'ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00016007.JPEG': 709,
 }
```

---

## HDF5

- A binary file format for storing large, complex datasets.

- Store data like a file system inside a file.

- Hierarchical: organizes data as groups and datasets

---

## HDF5

![](images/hdf5.png)

---

## PyArrow

- A Python library that provides tools for Apache Arrow – an in-memory columnar data

- Stores data as tables, arrays, and record batches

---

## PyArrow

![](images/pyarrow.png)


--- 

## Run examples

- The examples are in: 

    ```bash 
        imagenet_loaders.py # to create the H5 and pyarrow files  
        save_imagenet_files.py # to read the H5 and pyarrow files
    ```

- To create the h5 or pyarrow files, you can run the examples by launching 

    ```bash 
        sbatch run_save_file.sh
    ```
    
- To read those files, you can run:

    ```bash 
        run_loader.sh
    ```
