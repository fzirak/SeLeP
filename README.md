# SeLeP

## SeLeP: Learning Based Semantic Prefetching for Exploratory Database Workloads

We propose a learning-based semantic prefetcher which predicts the subsequent data accesses using actual data rather than the logical block addresses (LBA). The system dynamically clusters the blocks frequently accessed together in the same partitions.  The encoder-decoder LSTM prediction model gets sequence of query encodings and outputs the probability of each partitions being accessed in the next query.

For more details you can refer to our paper.

If you would like to use our code, please cite our paper:

Zirak, Farzaneh, Farhana Choudhury, and Renata Borovica-Gajic. "SeLeP: Learning Based Semantic Prefetching for Exploratory Database Workloads." arXiv preprint arXiv:2310.14666 (2023).

##

The clay-based partitioning is implemented as a separate Java project located in the 'Partitioning' folder. You can utilize the 'NewMain' class to extract the block IDs accessed by simple queries. For more complex queries you need  to access your Database Cache contents. The 'ServerMainThread' serves as the core for partitions, clustering blocks based on the workload received from clients.
To simplify the process, you have the flexibility to create partitions using any algorithms and input them into the SeLeP prefetcher.