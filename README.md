# T5-text-to-sql

**SETUP**

The setup is very simple.
we have used **setup.py** and **requirements.txt** as a wrapper around the different packages needed for this script.

on the terminal run

    pip install -e .

**TRAINING**

Training is done using our script **train_new.py**, 

Shown below is an example command line that we used for our training.

    python train_new.py    --model_name_or_path t5-base \
                           --source_prefix "translate to SQL: " \
                           --dataset_name spider  \
                           --output_dir t5_base_spider \
                           --per_device_train_batch_size=32  \
                           --per_device_eval_batch_size=32    \
                           --predict_with_generate True \
                           --num_train_epochs 15 \
                           --preprocessing_num_workers 8 \
                           --gradient_accumulation_steps 16 \
                           --eval_every_step 3

The most important args that are needed for our setup in order to train the model are:

**--model_name** <t5-small/t5-base/t5-large/t5-3b> this will fetch the respective model from the huggingface library.

**--source_prefix** <"text to SQL: > this is probably the most important arg for our transfer learning script.Since t5 is a transfer learning sequence to sequence model i needs a prefix to be added infront of the target ids to make the model learn about the type of learning it has to do. t5 is general enough to accomodate a variety of different transfer learning tasks.

**--output_dir** <t5_small_spider_train> this is the output directory where we save the model and checkpoint if neccesary.

**--dataset_name** <spider/cosql> We have used spider as our golden dataset. but the script can be used for cosql as well. We will need to create the gold_example.txt if we want to use cosql.


**EVALUATION**

The evaluation is done using **evaluation.py** which in turn calls **process_sql.py**.

In order to setup this step we need to do one additional setup for nltk dataset. Run the below cmd for the setup: 

    python -m nltk.downloader all

or if you are using a notebook/jupyter/google colab then use

    import nltk
    nltk.download("punkt")

unzipping the spider dataset is needed to run this evaluation script. The evaluation script needs to access the database schemas/db and its tables for the questions. We have included the zip but if its corrupted please use the below link for downloading the latest spider dataset.

Spider dataset can be found here:-
    https://yale-lily.github.io/spider

Run the below cmd for partial evaluation:

    python evaluation.py \
          --gold gold_example.txt \
          --pred pred_example.txt \
          --etype all \
          --db /content/drive/My\ Drive/spider/database \
          --table /content/drive/My\ Drive/spider/tables.json

The **gold_example.txt** contains all the 1034 golden queries from the spider dataset from huggingface library. Each line is a query corresponding to the spider validation datset split. for dexterity purposes we do not shuffle the validation set during our evaluation/prediction phase.

Here are some lines from the file just to give us an idea of what it contains.

    SELECT country FROM singer WHERE age  >  40 INTERSECT SELECT country FROM singer WHERE age  <  30	concert_singer

    SELECT name FROM stadium EXCEPT SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  =  2014	concert_singer

    SELECT name FROM stadium EXCEPT SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  =  2014	concert_singer

    SELECT T2.concert_name ,  T2.theme ,  count(*) FROM singer_in_concert AS T1 JOIN concert AS T2 ON T1.concert_id  =  T2.concert_id GROUP BY T2.concert_id	concert_singer

The **pred_example.txt** file will be created during training/evaluation and should be inside the **--output_dir** after your training has been completed.

This is used to then match up with the **gold_example.txt** to compute all the different metrices for evalauation.









