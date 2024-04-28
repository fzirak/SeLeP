# SeLeP

## SeLeP: Learning Based Semantic Prefetching for Exploratory Database Workloads

We propose a learning-based semantic prefetcher which predicts the subsequent data accesses using actual data rather than the logical block addresses (LBA). The system dynamically clusters the blocks frequently accessed together in the same partitions.  The encoder-decoder LSTM prediction model gets sequence of query encodings and outputs the probability of each partitions being accessed in the next query.

For more details you can refer to [our paper](https://arxiv.org/pdf/2310.14666).

If you would like to use our code, please cite our paper:

Zirak, Farzaneh, Farhana Choudhury, and Renata Borovica-Gajic. "SeLeP: Learning Based Semantic Prefetching for Exploratory Database Workloads." arXiv preprint arXiv:2310.14666 (2023).

##
## Steps to test SeLeP
1. **Install the requirements** - All required pachages and libraries are listed in the `requirements.txt` file and you can install them using:
    ```sh
    pip install -r requirements.txt
    ```
2. **Setup the Database** - Create your database in PostgreSQL.
2. **Set up the config file** (Configuration.Config) - add database connection information and set the configuration parameters values (they are set to the default values and you can left them as is).
3. **Prepare workload files** - The workload file should contain the following columns:

    | theTime | ClientIP | row | statement | resultBlock |
    | ------ | ------ | ------ | ------ | ------ | 
    03/23/2014 05:51:59 PM|140.1.2.0|10|select * from tbplasmiddna   where ngulwater < 0 |[tbplasmiddna_20, tbplasmiddna_18, tbplasmiddna_17, tbplasmiddna_19]
    
    The _resultBlock_ column, stores blocks accessed by the query statement. You can use `bid_getter.py` to get _resultBlock_ of a workload. This code, after clearing PostgreSQL and system cache, executes the query and checks the cache contents. To clear the caches, you need super user access previllage.


4. **Create partitions** - Run `partitioning_main.py` to create partitions using clay-based partitioning on a specific workload file. You can also use this code to find _resultPartitions_ for each query based on the latest set of partitions.
To simplify the process, you have the flexibility to create partitions using any algorithms and input them into the SeLeP prefetcher.
5. **Train SeLeP** - Run `selep_main.py` with `do_train=1`. When you run this code on a dataset for the first time, it will create block encodings and store them for later access. After gathering the block encodings, it generates model input and output and trains the encoder-decoder LSTM model. Lastly, it stores the model for future tests.
6. **Test SeLeP** - Set the test config (e.g. cache size, prefetch size, etc.) and run `selep_main.py` with `do_train=0`. It will load the model trained in the previous step and use it to make prediction on a given workload. 
7. **Test results** - All test results can be found in the Results folder and can be accessed to plot graphs.
8. (optional) **Test traditional prefetchers** - The traditional prefetchers explained in the paper are implemented and can be tested by running `main.py` file.
