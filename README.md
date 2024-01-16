# A Survey of Large Language Models Attribution [[ArXiv preprint](https://arxiv.org/abs/2311.03731)]

---

## 🌟 Introduction

Open-domain dialogue systems, driven by large language models, have changed the way we use conversational AI. 
However, these systems often produce content that might not be reliable. In traditional open-domain settings, the focus is mostly on the answer’s relevance or accuracy rather than evaluating whether the answer is attributed to the retrieved documents. A QA model with high accuracy may not necessarily achieve high attribution. 

Attribution refers to the capacity of a model, such as an LLM, to generate and provide evidence, often in the form of references or citations, that substantiates the claims or statements it produces. This evidence is derived from identifiable sources, ensuring that the claims can be logically inferred from a foundational corpus, making them comprehensible and verifiable by a general audience. The primary purposes of attribution include enabling users to validate the claims made by the model, promoting the generation of text that closely aligns with the cited sources to enhance accuracy and reduce misinformation or hallucination, and establishing a structured framework for evaluating the completeness and relevance of the supporting evidence in relation to the presented claims.


In this repository, we focus on unraveling the sources that these systems tap into for attribution or citation. We delve into the origins of these facts, their utilization by the models, the efficacy of these attribution methodologies, and grapple with challenges tied to ambiguous knowledge reservoirs, inherent biases, and the pitfalls of excessive attribution.

**✨ Work in progress. We would like to appreciate any contributions via PRs, issues from NLP community.**

---

## 1. Attribution Definition & Position Paper

*   [2021/07] **Rethinking Search: Making Domain Experts out of Dilettantes** *Donald Metzler et al. arXiv.* [[paper](https://arxiv.org/pdf/2105.02274.pdf)] 

     <!-- ```
      This position paper says "For example, for question answering tasks our envisioned model is able to synthesize a singleanswer that incorporates information from many documents in the corpus, and it will be able to support assertions in the answer by referencing supporting evidence in the corpus, much like a properly crafted Wikipedia entry supports each assertion of fact with a link to a primary source. This is just one of many novel tasks that this type of model has the potential to enable." 
      ```-->
    
 <!--*   [2023/03] **TRAK: Attributing Model Behavior at Scale** *Sung Min Park et al. arXiv.* [[paper](https://arxiv.org/abs/2303.14186)][[code](https://github.com/MadryLab/trak)]
     ```
      Attributing Model: trace model predictions back to training data. This paper introduces a data attribution method that is both effective and computationally tractable for large-scale, differentiable models.
      ```-->

  
*   [2021/12] **Measuring Attribution in Natural Language Generation Models.** *H Rashkin et al. CL.* [[paper](https://arxiv.org/pdf/2112.12870.pdf)]
      <!--```
      This paper presents a new evaluation framework entitled Attributable to Identified Sources (AIS) for assessing the output of natural language generation models.
      ```-->
*   [2022/11] **The attribution problem with generative AI** *Anna Rogers* [[blog](https://hackingsemantics.xyz/2022/attribution/)]
  
*   [2023/07] **Citation: A Key to Building Responsible and Accountable Large Language Models** *Jie Huang et al. arXiv.* [[paper](https://arxiv.org/pdf/2307.02185.pdf)] 
      <!--```
      This position paper embarks on an exploratory journey into the potential of integrating a citation mechanism within large language models, examining its prospective benefits, the inherent technical obstacles, and foreseeable pitfalls.
      ```-->

*   [2023/10] **Establishing Trustworthiness: Rethinking Tasks and Model Evaluation** *Robert Litschko et al. arXiv.* [[paper](https://arxiv.org/pdf/2310.05442.pdf)]

*  [2023/11] **Unifying Corroborative and Contributive Attributions in Large Language Models** *Theodora Worledge et al. NeurIPS ATTRIB Workshop 2023* [[paper](https://arxiv.org/pdf/2311.12233.pdf)]


## 2. Attribution Paper Before the Era of Large Language Models and Related Task

### 2.1 Fact Checking & Claim Verificication & Natural Language Inference

*   [2021/11] **The Fact Extraction and VERification (FEVER) Shared Task**  *James Thorne et al. EMNLP'18* [[paper](https://aclanthology.org/W18-5501v3.pdf)]

*   [2021/08] **A Survey on Automated Fact-Checking**  *Zhijiang Guo et al. TACL'22*  [[paper](https://arxiv.org/pdf/2108.11896.pdf)]

*   [2021/10] **Truthful AI: Developing and governing AI that does not lie** *Owain Evans et al. arXiv* [[paper](http://arxiv.org/abs/2110.06674)]

*   [2021/05] **Evaluating Attribution in Dialogue Systems: The BEGIN Benchmark** *Nouha Dziri et al. TACL'22* [[paper](https://aclanthology.org/2022.tacl-1.62/)] [[code](https://github.com/google/BEGIN-dataset)]

*   [2023/10] **Explainable Claim Verification via Knowledge-Grounded Reasoning with Large Language Models** *Haoran Wang et al. Findings of EMNLP'23* [[paper](https://arxiv.org/abs/2310.05253)]

*   [2022/07] **Improving Wikipedia Verifiability with AI** *Fabio Petroni et al. arXiv.* [[paper](https://arxiv.org/pdf/2207.06220.pdf)] [[code](https://github.com/facebookresearch/side)]

### 2.2 Feature Attribution and Interpretability of Models for NLP 
*   [2022/12] **Foveate, Attribute, and Rationalize: Towards Physically Safe and Trustworthy AI** *Alex Mei et al. findings of ACL'22* [[paper](https://aclanthology.org/2023.findings-acl.701.pdf)]

*   [2023/07] **Inseq: An Interpretability Toolkit for Sequence Generation Models** *Gabriele Sarti et al. ACL Demo'23* [[paper](https://aclanthology.org/2023.acl-demo.40)] [[library](https://github.com/inseq-team/inseq)]

*   [2023/10] **Quantifying the Plausibility of Context Reliance in Neural Machine Translation** *Gabriele Sarti et al. arXiv.* [[paper](https://arxiv.org/abs/2310.01188)]

### 2.3 Attribution in Mutli-modal Systems

* [2017/06] **A unified view of gradient-based attribution methods for Deep Neural Networks.** *Marco Ancona et al. arXiv.* [[paper](http://www.interpretable-ml.org/nips2017workshop/papers/02.pdf)]

* [2021/03] **Towards multi-modal causability with Graph Neural Networks enabling information fusion for explainable AI.** *Andreas Holzinger et al. arXiv.* [[paper](https://featurecloud.eu/wp-content/uploads/2021/03/Holzinger-et-al_2021_Towards-multi-model-causability.pdf)]

* [2023/03] **Retrieving Multimodal Information for Augmented Generation: A Survey.** *Ruochen Zhao et al. arXiv.* [[paper](https://arxiv.org/pdf/2303.10868.pdf)]

* [2023/07] **Improving Explainability of Disentangled Representations using Multipath-Attribution Mappings.** *Lukas Klein et al. arXiv.* [[paper](https://arxiv.org/pdf/2306.09035.pdf)]

* [2023/07] **Visual Explanations of Image-Text Representations via Mult-Modal Information Bottleneck Attribution.** *Ying Wang et al. arXiv.* [[paper](https://openreview.net/pdf?id=enSkaeByTE)]

* [2023/07] **MAEA: Multimodal Attribution for Embodied AI.** *Vidhi Jain et al. arXiv.* [[paper](https://arxiv.org/pdf/2307.13850.pdf)]

* [2023/10] **Rephrase, Augment, Reason: Visual Grounding of Questions for Vision-Language Models.** *Archiki Prasad et al. arXiv.* [[paper](https://arxiv.org/abs/2310.05861)] [[code](https://github.com/archiki/RepARe)]

### 2.4 Wiki

* [2019/11] **Transforming Wikipedia into Augmented Data for Query-Focused Summarization.** *Haichao Zhu et al. TASLP.* [[paper](https://arxiv.org/pdf/1911.03324.pdf)]

* [2023/04] **WebBrain: Learning to Generate Factually Correct Articles for Queries by Grounding on Large Web Corpus.** *Hongjin Qian et al. arXiv.* [[paper](https://arxiv.org/pdf/2304.04358.pdf)]

### 2.5 Model based Information Retrieval

* [2022/02] **Transformer Memory as a Differentiable Search Index** *Yi Tay et al. NeurIPS'22* [[paper](https://arxiv.org/pdf/2202.06991.pdf)]

### 2.6 Small Language Model

* [2023/08] **Optimizing Factual Accuracy in Text Generation through Dynamic Knowledge Selection** *Hongjin Qian et al. arxiv.* [[paper](https://arxiv.org/pdf/2308.15711.pdf)]


## 3. Sources of Attribution

### 3.1 Pre-training Data
* [2023/02] **The ROOTS Search Tool: Data Transparency for LLMs** *Aleksandra Piktus et al. arXiv.* [[paper](https://arxiv.org/pdf/2302.14035.pdf)]
* [2022/05] **ORCA: Interpreting Prompted Language Models via Locating Supporting Data Evidence in the Ocean of Pretraining Data** *Xiaochuang Han et al. arXiv.* [[paper](https://arxiv.org/pdf/2205.12600.pdf)]

* [2022/05] **Understanding In-Context Learning via Supportive Pretraining Data** *Xiaochuang Han et al. arXiv.* [[paper](https://arxiv.org/pdf/2306.15091.pdf)]

* [2022/07] [link the fine-tuned LLM to its pre-trained base model]  **Matching Pairs: Attributing Fine-Tuned Models to their Pre-Trained Large Language Models** *Myles Foley et al. ACL 2023.* [[paper](https://aclanthology.org/2023.acl-long.410.pdf)]

### 3.2 Out-of-model Knowledge and Retrieval-based Question Answering & Knowledge-Grounded Dialogue

* [2021/04] **Retrieval augmentation reduces hallucination in conversation** *Kurt Shuster et al. arXiv.* [[paper](https://arxiv.org/pdf/2104.07567.pdf)]

* [2020/07] **Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering** *Gautier Izacard et al. arXiv.* [[paper](https://arxiv.org/pdf/2007.01282.pdf)]

* [2021/12] **Improving language models by retrieving from trillions of tokens** *Sebastian Borgeaud et al. arXiv.* [[paper](https://arxiv.org/pdf/2112.04426.pdf)]

* [2022/12] **Rethinking with Retrieval: Faithful Large Language Model Inference**  *Hangfeng He et al. arXiv.* [[paper](https://arxiv.org/pdf/2301.00303.pdf)] 

## 4. Datasets for Attribution
* [2022/12] **CiteBench: A benchmark for Scientific Citation Text Generation** *Martin Funkquist et al. arXiv.* [[paper](https://arxiv.org/abs/2212.09577)]

* [2023/04] **WebBrain: Learning to Generate Factually Correct Articles for Queries by Grounding on Large Web Corpus** *Hongjing Qian et al. arXiv.* [[paper](https://arxiv.org/pdf/2304.04358.pdf)] [[code](https://github.com/qhjqhj00/WebBrain)]

* [2023/05] **Enabling Large Language Models to Generate Text with Citations** *Tianyu Gao et al. arXiv.* [[paper](https://arxiv.org/pdf/2305.14627.pdf)] [[code](https://github.com/princeton-nlp/ALCE)]
   <!--```
   This paper proposes ALCE dataset, which collects a diverse set of questions and retrieval corpora and requires building end-to-end systems to retrieve supporting evidence and generate answers with citations.
   ```-->

* [2023/07] **HAGRID: A Human-LLM Collaborative Dataset for Generative Information-Seeking with Attribution** *Ehsan Kamalloo et al. arXiv.* [[paper](https://arxiv.org/pdf/2307.16883.pdf)] [[code](https://github.com/project-miracl/hagrid)]
   <!--```
   This paper introduces the HAGRID dataset for building end-to-end generative information-seeking models that are capable of retrieving candidate quotes and generating attributed explanations.
   ```-->
* [2023/09] **EXPERTQA : Expert-Curated Questions and Attributed Answers** *Chaitanya Malaviya et al. arXiv.* [[paper](https://arxiv.org/pdf/2309.07852.pdf)] [[code](https://github.com/chaitanyamalaviya/expertqa)]
   <!--```
   This paper introduces the EXPERTQA, a high-quality long-form QA dataset with 2177 questions spanning 32 fields, along with verified answers and attributions for claims in the answers. 
   ```-->
* [2023/11] **SEMQA: Semi-Extractive Multi-Source Question Answering** *Tal Schuster et al. arXiv.* [[paper](https://arxiv.org/pdf/2311.04886.pdf)] [[code](https://github.com/google-research-datasets/QuoteSum)]
   <!--```
   This paper  introduce a new QA task, Semi-extractive Multi-source QA (SEMQA), for answering multi-answer questions by summarizing multiple diverse sources in a semi-extractive fashion, which requires models to output a comprehensive answer, while mixing factual quoted spans—copied verbatim from given input sources—and non-factual free-text connectors that glue these spans together into a single cohesive passage.
    ```-->

## 5. Approaches to Attribution

### 5.1 Direct Generated Attribution
*   [2023/05] **"According to ..." Prompting Language Models Improves Quoting from Pre-Training Data** *Orion Weller et al. arXiv.*  [[paper](https://arxiv.org/abs/2305.13252)]
      <!--```
      This paper proposes according-to prompting to directing LLMs to ground responses against previously observed text, and propose QUIP-Score to measure the extent to which model-produced answers are directly found in underlying text corpora.
      ```-->

*  [2023/07] **Credible Without Credit: Domain Experts Assess Generative Language Models** *Denis Peskoff et al. ACL 2023.* [[paper](https://aclanthology.org/2023.acl-short.37/)]

*   [2023/09] **ChatGPT Hallucinates when Attributing Answers** *Guido Zuccon et al. arXiv.* [[paper](https://arxiv.org/abs/2309.09401)]
      <!--```
      This paper suggests that ChatGPT provides correct or partially correct answers in about half of the cases (50.6% of the times), but its suggested references only exist 14% of the times. In thoses referenced answers, the reference often does not support the claims ChatGPT attributes to it.
      ```-->
*    [2023/09] **Towards Reliable and Fluent Large Language Models: Incorporating Feedback Learning Loops in QA Systems** *Dongyub Lee et al. arXiv.* [[paper](https://arxiv.org/pdf/2309.06384.pdf)]

*    [2023/09] **Retrieving Evidence from EHRs with LLMs: Possibilities and Challenges** *Hiba Ahsan et al. arXiv.* [[paper](https://arxiv.org/pdf/2309.04550.pdf)]

* [2023/10] **Learning to Plan and Generate Text with Citations** *Annoymous et al. OpenReview, ICLR 2024* [[paper](https://openreview.net/forum?id=6NEJ0ReNzr)]

* [2023/10] **1-PAGER: One Pass Answer Generation and Evidence Retrieval** *Palak Jain et al. arxiv* [[paper](https://arxiv.org/pdf/2310.16568v1.pdf)]



### 5.2 Retrieval-then-Answering

*  [2023/04] **Search-in-the-Chain: Towards the Accurate, Credible and Traceable Content Generation for Complex Knowledge-intensive Tasks** *Shicheng Xu et al. arXiv.* [[paper](https://arxiv.org/pdf/2304.14732.pdf)]

* [2023/05] **Mitigating Language Model Hallucination with Interactive Question-Knowledge Alignment** *Shuo Zhang et al. arXiv.* [[paper](https://arxiv.org/pdf/2305.13669.pdf)]

* [2023/03] **SmartBook: AI-Assisted Situation Report Generation** *Revanth Gangi Reddy et al. arXiv.* [[paper](https://arxiv.org/abs/2303.14337)]

* [2023/10] **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection** *Akari Asai et al. arXiv.* [[paper](https://arxiv.org/abs/2310.11511)] [[homepage](https://selfrag.github.io/)]

* [2023/11] **LLatrieval: LLM-Verified Retrieval for Verifiable Generation** *Xiaonan Li et al. arXiv.* [[paper](https://arxiv.org/pdf/2311.07838.pdf)] [[code](https://github.com/BeastyZ/LLM-Verified-Retrieval)]

* [2023/11] **Effective Large Language Model Adaptation for Improved Grounding** *Xi Ye et al. arXiv.* [[paper](https://arxiv.org/pdf/2311.09533.pdf)]

* [2024/1] **Towards Verifiable Text Generation with Evolving Memory and Self-Reflection** *Hao Sun et al. arXiv.* [[paper](https://arxiv.org/pdf/2312.09075.pdf)]

* [2023/5] **WikiChat: Stopping the Hallucination of Large Language Model Chatbots by Few-Shot Grounding on Wikipedia** *Sina Semnani et al. Findings of EMNLP'23** [[paper]](https://arxiv.org/abs/2305.14292) [[code]](https://github.com/stanford-oval/WikiChat) [[demo]](https://wikichat.genie.stanford.edu)

### 5.3 Post-Generation Attribution

*  [2022/10] **RARR: Researching and Revising What Language Models Say, Using Language Models** *Luyu Gao et al. arXiv.* [[paper](https://arxiv.org/abs/2210.08726)]

*  [2023/04] **The Internal State of an LLM Knows When its Lying** *Amos Azaria et al. arXiv.* [[paper](http://arxiv.org/abs/2304.13734)]
      <!--```
      This paper utilizes the LLM's hidden layer activations to determine the veracity of statements by a classifier receiveing as input the activation values from the LLM for each of the statements in the dataset.
      ```-->
*  [2023/05] **Do Language Models Know When They're Hallucinating References?** *Ayush Agrawal et al. arXiv.* [[paper](http://arxiv.org/abs/2305.18248)]

*  [2023/05] **Complex Claim Verification with Evidence Retrieved in the Wild** *Jifan Chen et al. arXiv.*  [[paper](https://arxiv.org/abs/2305.11859)][[code](https://github.com/jifan-chen/fact-checking-via-raw-evidence)]
      <!--```
      This paper proposes a pipeline(claim decomposition, multi-granularity evidence retrieval, claim-focused summarization) to improve veracity judgments.
      ```-->
*  [2023/06] **Retrieving Supporting Evidence for LLMs Generated Answers** *Siqing Huo et al. arXiv.*  [[paper](https://arxiv.org/pdf/2309.11392.pdf)]
      <!--```
      This paper proposes a two-step verification. The LLM's answer and the retrieved document queried by question and LLM's answer are compared by LLM, checking whether the LLM's answer is hallucinated.
      ```-->



### 5.4 Attribution Systems & End-to-End Attribution Models

* [2022/03] **LaMDA: Language Models for Dialog Applications.** *Romal Thoppilan et al. arXiv.* [[paper](https://arxiv.org/pdf/2201.08239.pdf)]

*  [2022/03] **WebGPT: Browser-assisted question-answering with human feedback.** *Reiichiro Nakano, Jacob Hilton, Suchir Balaji et al. arXiv.*[[paper](https://arxiv.org/pdf/2112.09332.pdf)]

*  [2022/03] **GopherCite - Teaching language models to support answers with verified quotes.**  *Jacob Menick  et al. arXiv.* [[paper](http://arxiv.org/abs/2203.11147)]

*  [2022/09] **Improving alignment of dialogue agents via targeted human judgements.**  *Amelia Glaese  et al. arXiv.* [[paper](https://arxiv.org/pdf/2209.14375.pdf)]

* [2023/05] **WebCPM: Interactive Web Search for Chinese Long-form Question Answering.** *Yujia Qin et al. arXiv.*  [[paper](https://arxiv.org/pdf/2305.06849.pdf)]




## 6. Attribution Evaluation





* [2022/07] **Improving Wikipedia Verifiability with AI** *Fabio Petroni et al. arXiv.* [[paper](https://arxiv.org/pdf/2207.06220.pdf)]

*   [2022/12] **Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models.** *B Bohnet et al. arXiv.* [[paper](https://arxiv.org/pdf/2212.08037.pdf)] [[code](https://github.com/google-research-datasets/Attributed-QA)]
*   [2023/04] **Evaluating Verifiability in Generative Search Engines** *Nelson F. Liu et al. arXiv.* [[paper](http://arxiv.org/abs/2304.09848)] [[annonated data](https://github.com/nelson-liu/evaluating-verifiability-in-generative-search-engines)] 
*   [2023/05] **WICE: Real-World Entailment for Claims in Wikipedia** *Ryo Kamoi et al. arXiv.* [[paper](https://arxiv.org/pdf/2303.01432.pdf)]
*   [2023/05] **Evaluating and Modeling Attribution for Cross-Lingual Question Answering** *Benjamin Muller et al. arXiv.* [[paper](https://arxiv.org/abs/2305.14332)]
*   [2023/05] **FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation** *Sewon Min et al. arXiv.* [[paper](https://arxiv.org/abs/2305.14251)] [[code](https://github.com/shmsw25/FActScore)]


*   [2023/05] **Automatic Evaluation of Attribution by Large Language Models.** *X Yue et al. arXiv.* [[paper](https://arxiv.org/pdf/2305.06311.pdf)] [[code](https://github.com/OSU-NLP-Group/AttrScore)]
      <!--```
      This paper investigate the automatic evaluation of attribution by LLMs - AttributionScore, by providing a definition of attribution and then explore two approaches for automatic evaluation. The results highlight both promising signals as well as remaining challenges for the automatic evaluation of attribution.
      ```-->
*   [2023/07] **FacTool: Factuality Detection in Generative AI -- A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios** *I-Chun Chern et al. arXiv.* [[paper](https://arxiv.org/abs/2307.13528v2)][[code](https://github.com/GAIR-NLP/factool)]

*   [2023/09] **Quantifying and Attributing the Hallucination of Large Language Models via Association Analysis** *Li Du et al. arXiv.* [[paper](https://arxiv.org/pdf/2309.05217v1.pdf)]

*   [2023/10] **Towards Verifiable Generation: A Benchmark for Knowledge-aware Language Model Attribution** *Xinze Li et al. arXiv.* [[paper](https://arxiv.org/pdf/2310.05634.pdf)]
   <!--```
   This paper defines a new task of Knowledge-aware Language Model Attribution (KaLMA), and builds a dataset in biography domain BioKaLMA via a well-designed evolutionary question generation strategy.
   ```-->

* [2023/10] **Understanding Retrieval Augmentation for Long-Form Question Answering** *Hung-Ting Chen et al. arXiv.* [[paper](https://arxiv.org/pdf/2310.12150.pdf)]

* [2023/11] **Enhancing Medical Text Evaluation with GPT-4** *Yiqing Xie et al. arXiv.* [[paper](https://arxiv.org/pdf/2311.09581.pdf)]


## 7. Limitations, Future Directions and Challenges in Attribution

      a. hallucination of attribution i.e. does attribution faithfully to its content?
      b. Inability to attribute parameter knowledge of model self.
      c. Validity of the knowledge source - source trustworthiness. Faithfulness ≠ Factuality
      d. Bias in attribution method
      e. Over-attribution & under-attribution
      f. Knowledge conflict

--- 



## Cite

```
@misc{li2023llmattribution,
      title={A Survey of Large Language Models Attribution}, 
      author={Dongfang Li and Zetian Sun and Xinshuo Hu and Zhenyu Liu and Ziyang Chen and Baotian Hu and Aiguo Wu and Min Zhang},
      year={2023},
      eprint={2311.03731},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      howpublished={\url{https://github.com/HITsz-TMG/awesome-llm-attributions}},
}
```
***For finding survey of hallucination please refer to:***

- Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models
- Cognitive Mirage: A Review of Hallucinations in Large Language Models
- A Survey of Hallucination in Large Foundation Models
## Project Maintainers & Contributors
- [Dongfang Li](http://crazyofapple.github.io/)
- Zetian Sun
- Xinshuo Hu
- Zhenyu Liu

