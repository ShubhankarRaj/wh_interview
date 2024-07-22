---
base_model: sentence-transformers/all-MiniLM-L6-v2
datasets: []
language: []
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:525
- loss:MultipleNegativesRankingLoss
- dataset_size:526
widget:
- source_sentence: how does the pick-up store option on flipkart work?
  sentences:
  - in credit card emis, banks charge an interest rate based on the emi plan chosen
    whereas bajaj finserv emi card holders are not charged any interest on the emi.
    unlike other credit card based emi options, you don't need to have a credit card
    to avail emi plans. the emi payments are deducted by bajaj finserv from your bank
    account linked to the bajaj finserv emi card.
  - 'with the pick-up store option, you can choose to pick up your shipment from select
    pick-up stores at your convenience. all you need to do is choose the pick-up store
    option at the time of placing your order and you can walk into the pick-up location
    to collect your order once it reaches there.

    you''ll also receive an email and sms once the shipment has reached the pick-up
    store.'
  - '


    supercoin pay on flipkart platform enables all customers who have more than 9
    supercoins in their flipkart account to redeem their supercoins to avail discounts
    while buying a product. as part of supercoin pay, customers can avail discounts
    by redeeming supercoins and pay using a prepaid payment mode provided by participating
    banks. however, the redemption of supercoins will depend on the value of the product
    and terms and conditions of the paticipating bank or the payment mode.


    '
- source_sentence: how can i become a seller and start selling products on flipkart?
  sentences:
  - no, only twid (innotarget fashalot retail private limited) or qr codes associated
    with supercoin pay feature can be scanned to complete payment for an order using
    supercoin pay feature.
  - to become a seller on flipkart, register here with your details.
  - please note, once you've completed the purchase on flipkart, the brand authorised
    dealer will not market any other products. however, they may recommend accessories
    that may be suitable for your vehicle.
- source_sentence: can i redeem my 'on the way' supercoins to avail discounts on product?
  sentences:
  - 'supercoin zone is a section where third-party ''supercoins rewards'' can be availed
    by redeeming the earned supercoins.

    to know more, visit the supercoin zone here '
  - '


    please note, the 1st and the 4th services are free of cost. any other services
    may be chargeable. for more details, kindly get in touch with the brand at 8291939393.


    '
  - 'you can redeem ''credited'' supercoins to avail discounts for select products
    on flipkart.

     

    to know more about the flipkart plus program click herefor more information on
    supercoins click here'
- source_sentence: for how long is a payment code as a part of supercoin pay valid
    once it is generated?
  sentences:
  - '





    the credit limit varies based on the type of bajaj finserv emi card you hold.

    for more details, please check bajaj finserv support faqs here: https://www.bajajfinserv.in/insta-emi-card

    you can also contact bajaj finserv at 08698010101 for support.





    '
  - '


    you can view the details of your membership by following these simple steps:

    for app- log in to your account- click on flipkart plus logo

    for website- log in to your account- click on ''my account''- go to ''flipkart
    plus zone''

    to know more about the program click here

    for more information on supercoins click here

     


    '
  - a payment code generated will be valid for 24 hours hours.
- source_sentence: will my shipment be safe at the pickup store?
  sentences:
  - flipkart axis bank credit card is a credit card issued by axis bank in association
    with flipkart internet pvt ltd. (flipkart) and mastercard. every purchase that
    you make on flipkart.com and outside flipkart.com will earn you unlimited cashback
    that will be credited to your credit card account. you can use these cashback
    for your future purchases.please note that a joining/annual fee of rs. 500 is
    applicable unless it is acquired under the first year free offer.
  - currently, there is no option to club orders from different sellers to be delivered
    together as sellers could be located in different locations and the delivery timelines
    would vary based on their partnered courier service providers. to ensure your
    items reach you at the earliest, each seller ships their products as per their
    individual timelines.
  - rest assured, your shipment is safe with our trusted partners at all pickup stores.
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 8b3219a92973c328a8e22fadcfa821b5dc75636a -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'will my shipment\xa0be safe at the pickup store?',
    'rest assured, your shipment\xa0is safe with our trusted partners at all pickup stores.',
    'currently, there is no option to club orders from different sellers to be delivered together as sellers could be located in different locations and the delivery timelines would vary based on their partnered courier service providers. to ensure your items reach you at the earliest, each seller ships their products as per their individual timelines.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 526 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                          |
  |:--------|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                              |
  | details | <ul><li>min: 7 tokens</li><li>mean: 19.6 tokens</li><li>max: 67 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 58.54 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                  | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
  |:------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>i want to update my email id/phone number for the booking that i've just made, how do i do it?</code> | <code>you can get in touch with the airline to update your email id/phone number.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
  | <code>can i redeem my 'on the way' supercoins to avail discounts on product?</code>                         | <code>you can redeem 'credited' supercoins to avail discounts for select products on flipkart.<br> <br>to know more about the flipkart plus program click herefor more information on supercoins click here</code>                                                                                                                                                                                                                                                                                                                                                                                                                                          |
  | <code>how do i find the offers in supercoin zone?</code>                                                    | <code><br><br><br><br><br><br><br><br>you can find flipkart plus related offers by following these simple steps:<br>- login to your flipkart account- visit 'my accounts' - click on 'supercoin zone' and look for the 'offers' section- click on 'claim offer' to avail the offer as per the terms and conditions. the 'supercoins rewards' code will be available in the 'claimed offers' section if you wish to use it in future. you will also receive an email with the details at your registered email address.<br>to know more about the program click here<br>for more information on supercoins click here<br><br><br><br><br><br><br><br></code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 3.0.1
- Transformers: 4.42.4
- PyTorch: 2.3.1+cu121
- Accelerate: 0.32.1
- Datasets: 2.20.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply}, 
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->