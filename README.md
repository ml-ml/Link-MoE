# Link-MoE

1. Generate Prediction Scores for Base Models
   
	Please follow https://github.com/Juanhui28/HeaRT/tree/master to get the prediction scores for 10 seeds in existing setting. For example, for the gcn prediction scores of ogbl-collab dataset:
	```
	cd benchmarking/exist_setting_ogb
	python main_gnn_ogb.py  --use_valedges_as_input  --data_name ogbl-collab  --gnn_model GCN --hidden_channels 256 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3 --epochs 9999 --kill_cnt 100  --batch_size 65536 --save --outupt_dir ~/prediction_socres/collab/gcn
	```
  	The prediction score of gcn will be saved in the following format:
  	```
	{
		'pos_valid_score': pos_valid_pred,
		'neg_valid_score': neg_valid_pred,
		'pos_test_score': pos_test_pred,
		'neg_test_score': neg_test_pred,
		'node_emb': x1,
		'node_emb_with_valid_edges': x2
   }
  	```
2. Generate Heuristic Features
   
	Please follow https://github.com/Juanhui28/HeaRT/tree/master to get CN, AA, RA, Katz, PPR, Shorteset Path Length for each dataset. For example, for AA feature of ogbl-collab dataset:
	```
 	cd benchmarking/exist_setting_ogb
 	python main_heuristic_ogb.py --data_name ogbl-collab --use_heuristic AA --use_valedges_as_input --output_dir ~/heuristics/
 	```
 	Please modify the save_path in Line 252 as follows,
	```	
 	save_path = args.output_dir + args.data_name.split('-')[1] + '/' + args.use_heuristic
 	```
   	The AA feature will be saved in the following format:
  	```
	{'pos_test_score': [], 'neg_test_score': [], 'pos_valid_score': [], 'neg_valid_score': []}
  	```
3. Run the Codes
   
	ogbl-collab
	```
	python main.py --device 2 --use_valedges_as_input --data_name ogbl-collab --name collab --l2 0 --lr 0.001 --dropout 0 --num_layers 2 --hidden_channels 64 --score_number 0 --num_layers_predictor 1 --ncnc --neognn --buddy --mlp --n2v --seal --gcn --ncn --use_feature --use_degree --use_cn --use_sp --use_aa --use_ra --use_katz --use_ppr --runs 3 --end_epochs 800 --ratio 0.8 --train_batch_size 60048 --test_batch_size 100000 --kill_cnt 2000 
	```
	oglb-ppa
	```
	python main.py --device 2 --ratio 0.8 --data_name ogbl-ppa --name ppa --l2 0--lr 0.0001 --dropout 0 --num_layers 3 --hidden_channels 64  --score_number 0 --num_layers_predictor 1 --ncnc --neognn --buddy --mlp --n2v --seal --gcn --ncn --use_feature --use_degree --use_cn --use_sp --use_aa --use_ra --use_katz --use_ppr --runs 3 --end_epochs 500 --train_batch_size 50 --test_batch_size 60048 
	 ```
	ogbl-citation2
	```
 	python main.py --device 2 --ratio 0.8 --data_name ogbl-citation2 --name citation2 --l2 0 --lr 0.001 --dropout 0 --num_layers 2 --hidden_channels 64  --score_number 0 --num_layers_predictor 1 --ncnc --neognn --buddy --mlp --n2v --seal --gcn --ncn --use_feature --use_degree --use_cn --use_aa --use_ra --use_katz --runs 3 --train_batch_size 300 --test_batch_size 60048 --end_epochs 30 --kill_cnt 2000 
	```
     
	When running the codes in Step 1 and Step 2, please follow the provided parameters in 'https://github.com/Juanhui28/HeaRT/tree/master/scripts/hyperparameters/existing_setting_ogb'. 

	The score_number in Step 3 means the prediction results of base models in different seeds.
