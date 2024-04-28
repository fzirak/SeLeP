# SeLeP

## SeLeP: Learning Based Semantic Prefetching for Exploratory Database Workloads

We propose a learning-based semantic prefetcher which predicts the subsequent data accesses using actual data rather than the logical block addresses (LBA). The system dynamically clusters the blocks frequently accessed together in the same partitions.  The encoder-decoder LSTM prediction model gets sequence of query encodings and outputs the probability of each partitions being accessed in the next query.

For more details you can refer to [our paper](https://arxiv.org/pdf/2310.14666).

If you would like to use our code, please cite our paper:

Zirak, Farzaneh, Farhana Choudhury, and Renata Borovica-Gajic. "SeLeP: Learning Based Semantic Prefetching for Exploratory Database Workloads." arXiv preprint arXiv:2310.14666 (2023).

##
## Steps to test SeLeP
1. **Install the requirements** - All necessary packages and libraries are listed in the requirements.txt file. You can install them using the following command:
    ```sh
    pip install -r requirements.txt
    ```
2. **Setup the Database** - Create your database in PostgreSQL.
3. **Set up the config file** (Configuration/Config.py) - Provide the database connection details and set the configuration parameters values. The configuration parameters are set to default values and can be left unchanged.
4. **Prepare workload files** - The workload file should contain the following columns:

    | theTime | ClientIP | row | statement | resultBlock |
    | ------ | ------ | ------ | ------ | ------ | 
    03/23/2014 05:51:59 PM|140.1.2.0|10|select * from tbplasmiddna   where ngulwater < 0 |[tbplasmiddna_20, tbplasmiddna_18, tbplasmiddna_17, tbplasmiddna_19]
    
    The _resultBlock_ column, stores blocks accessed by the query statement. You can use `bid_getter.py` to get _resultBlock_ of a workload. This script executes the query after clearing the PostgreSQL and system cache and then checks the cache contents. Superuser access privileges are required to clear the caches.
    
5. **Create partitions** - Run `partitioning_main.py` to create partitions using clay-based partitioning on a specific workload file. Additionally, you can utilize this code to determine the _resultPartitions_ for each query based on the most recent set of partitions.
To simplify the process, you have the flexibility to create partitions using any algorithms and input them into the SeLeP prefetcher.
6. **Train SeLeP** - Run `selep_main.py` with the parameter do_train=1. When you run this code on a dataset for the first time, it will generate block encodings and save them for future use. Once the block encodings are collected, the code will generate model input and output and proceed to train the encoder-decoder LSTM model. Finally, it will store the trained model for future testing.
7. **Test SeLeP** - Configure the test settings, such as cache size and prefetch size, and then execute `selep_main.py` with the parameter `do_train=0`. This will load the model trained in the previous step and utilize it to make predictions on a provided workload.
8. **Test results** - All test results can be found in the Results folder and can be accessed to generate plots.
9. (optional) **Test traditional prefetchers** - The traditional prefetchers described in the paper have been implemented and can be tested by running the `main.py` file.