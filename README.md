# Code for Multimodal Bandits: Regret Lower Bounds and
Optimal Algorithms

This repository contains the Python code used for the runtime and regret experiments in the paper. 

## Code Structure
➤ The flags RUNTIME_EXPERIMENT, RUNTIME_IMPROVED_DP_EXPERIMENT, REGRET_EXPERIMENT can be set to True to run the experiments of Section 6 and Appendix E.8.
➤ synthetic experiment/ contains a main file (synthetic_experiment.py) that outputs a .pkl file and an auxiliary file (latex_table_from_pkl.py) that generates a LaTeX table from the .pkl file, summarizing the results of the experiments.  
➤  realdata experiment/ contains the .py files necessary to run the experiments on real-world datasets: encode/ contains the code used to convert the dataset queries into embeddings for each LLM used, all_metrics.py contains an implementation of the GHC algorithm with multiple distance metrics. faiss.py contains an implementation of the GHC algorithm using the FAISS library, which significantly speeds up computation. run_ETC.py and run_k_means.py contain an implementation of the ETC and sequential k-means algorithms respectively.   


<!-- LICENSE -->
## License

<a href="https://choosealicense.com/licenses/mit/">MIT</a> License.



<!--- ## 📬 Contact
Feel free to contact the authors:
William Réveillard wilrev@kth.se
Richard Combes richard.combes@centralesupelec.fr
 -->
