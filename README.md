# SI-GuidedProject-587985-1697708048

Adversarial Defense using AutoEncoder, Block Switching & GradCAM

This system majorly focuses on static image input and defence architecture. Following are the characteristics of the proposed model:
Combination of two models to effectively defend both Black box and White Box attack.
Randomization method acts as a backup for filtration performed by auto-encoder there by increasing the robustness of the proposed model.
Grad-CAM allows the model to predict the highlighted important region based on classification.

Installation
Prerequisites :

    1. Anaconda ( If using Python Notebook to run )
    2. Python
Download all dataset and necessary code

!git clone https://github.com/smartinternz02/SI-GuidedProject-587985-1697708048/tree/main

Install libraries using following commands
!pip install tensorflow
!pip install torch
!pip install keras
!pip install sklearn
!pip install cleverhans
!pip install glob
!pip install cv2
To run project runu the following command

python source_code.py

Dataset :
It is subset of Imagenet Dataset
Source : ImageNet 
           

Architecture :
        
ML Models Used :
   1. Resnet
   2. MobileNetV2
   3. Auto-Encoder
   4. DenseNet
Adversarial Attack :
Fast Gradient Sign Method(FGSM) - Goodfellow, I. J., Shlens, J., and Szegedy, C. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014b.
   def create_adversarial_pattern(input_image, input_label):
     with tf.GradientTape() as tape:
       tape.watch(input_image)
       prediction = pretrained_model(input_image)
       loss = loss_object(input_label, prediction)
     gradient = tape.gradient(loss, input_image)
     signed_grad = tf.sign(gradient)
     return signed_grad

Defense Architecture :
Module 1 : Auto Encoder
Auto-encoders can be used for Noise Filtering purpose. By feeding them noisy data as inputs and clean data as outputs, it’s possible to make them remove noise from the input image. This way, auto-encoders can serve as denoisers.
Module 2 : Randomisation
Switching block in this experiment consists of multiple channels. Each regular model is split into a lower part, containing all convolutional layer. lower parts are again combined to form single output providing parallel channels of block switching while the other parts are discarded. These models tend to have similar characteristics in terms of classification accuracy and robustness, yet different model parameters due to random initialization and stochasticity in the training process
Module 3 : Grad-CAM
Grad-CAM ( Activation Maps )uses the gradients of any target concept (say logits for “dog” or even a caption), flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.”
Auto Encoder :
Input – Adversarial image
Output – Clean image after removing noise
Auto-encoders can be used for filtration purpose.
It is possible for them to remove adversarial noise from an input image.


  

Grad CAM :
Input – Gradients of convolutional layer
Output – Activation/heat map
Grad-CAM are activation maps which generate highlights on the classified image to uncover important regions.
As you can see in the below example gradcam helps us to figure out why the image is predicted as corn instead of dog.

Publication
Title : An integrated Auto Encoder-Block Switching defense approach to prevent adversarial attacks 
Abstract : According to the recent studies, the vulnerability of state of the art Neural Networks to adversarial input samples has increased drastically. Neural network is an intermediate path or technique by which a computer learns to perform tasks using Machine learning algorithms. Machine Learning and Artificial Intelligence model has become fundamental aspect of life, such as self-driving cars, smart home devices, so any vulnerability is a significant concern. The smallest input deviations can fool these extremely literal systems and deceive their users as well as administrator into precarious situations. This article proposes a defense algorithm which utilizes the combination of an auto- encoder and block-switching architecture. Auto-coder is intended to remove any perturbations found in input images whereas block switching method is used to make it more robust against White-box attack. Attack is planned using FGSM model, and the subsequent counter-attack by the proposed architecture will take place thereby demonstrating the feasibility and security delivered by the algorithm



0. Toolkits
   
2.	OpenAttack: An Open-source Textual Adversarial Attack Toolkit. Guoyang Zeng, Fanchao Qi, Qianrui Zhou, Tingji Zhang, Bairu Hou, Yuan Zang, Zhiyuan Liu, Maosong Sun. ACL-IJCNLP 2021 Demo. [website] [doc] [pdf]
3.	TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP. John Morris, Eli Lifland, Jin Yong Yoo, Jake Grigsby, Di Jin, Yanjun Qi. EMNLP 2020 Demo. [website] [doc] [pdf]
4.	SeqAttack: On Adversarial Attacks for Named Entity Recognition. Walter Simoncini, Gerasimos Spanakis. EMNLP 2021 Demo. [website] [pdf]
   
1. Survey Papers
   
1.	Measure and Improve Robustness in NLP Models: A Survey. Xuezhi Wang, Haohan Wang, Diyi Yang. NAACL 2022. [pdf]
2.	Towards a Robust Deep Neural Network in Texts: A Survey. Wenqi Wang, Lina Wang, Benxiao Tang, Run Wang, Aoshuang Ye. TKDE 2021. [pdf]
3.	Adversarial Attacks on Deep Learning Models in Natural Language Processing: A Survey. Wei Emma Zhang, Quan Z. Sheng, Ahoud Alhazmi, Chenliang Li. ACM TIST 2020. [pdf]
4.	Adversarial Attacks and Defenses in Images, Graphs and Text: A Review. Han Xu, Yao Ma, Hao-chen Liu, Debayan Deb, Hui Liu, Ji-liang Tang, Anil K. Jain. International Journal of Automation and Computing 2020. [pdf]
5.	Analysis Methods in Neural Language Processing: A Survey. Yonatan Belinkov, James Glass. TACL 2019. [pdf]
   
2. Attack Papers
   
Each paper is attached to one or more following labels indicating how much information the attack model knows about the victim model: gradient (=white, all information), score (output decision and scores), decision (only output decision) and blind (nothing)

2.1 Sentence-level Attack

1.	Using Adversarial Attacks to Reveal the Statistical Bias in Machine Reading Comprehension Models. Jieyu Lin, Jiajie Zou, Nai Ding. ACL-IJCNLP 2021. blind [pdf]
2.	Grey-box Adversarial Attack And Defence For Sentiment Classiﬁcation. Ying Xu, Xu Zhong, Antonio Jimeno Yepes, Jey Han Lau. NAACL-HLT 2021. gradient [pdf] [code]
3.	Generating Syntactically Controlled Paraphrases without Using Annotated Parallel Pairs. Kuan-Hao Huang and Kai-Wei Chang. EACL 2021. [pdf] [code]
4.	CAT-Gen: Improving Robustness in NLP Models via Controlled Adversarial Text Generation. Tianlu Wang, Xuezhi Wang, Yao Qin, Ben Packer, Kang Lee, Jilin Chen, Alex Beutel, Ed Chi. EMNLP 2020. score [pdf]
5.	T3: Tree-Autoencoder Constrained Adversarial Text Generation for Targeted Attack. Boxin Wang, Hengzhi Pei, Boyuan Pan, Qian Chen, Shuohang Wang, Bo Li. EMNLP 2020. gradient [pdf] [code]
6.	Adversarial Attack and Defense of Structured Prediction Models. Wenjuan Han, Liwen Zhang, Yong Jiang, Kewei Tu. EMNLP 2020. blind [pdf] [code]
7.	MALCOM: Generating Malicious Comments to Attack Neural Fake News Detection Models. Thai Le, Suhang Wang, Dongwon Lee. ICDM 2020. gradient [pdf] [code]
8.	Improving the Robustness of Question Answering Systems to Question Paraphrasing. Wee Chung Gan, Hwee Tou Ng. ACL 2019. blind [pdf] [data]
9.	Trick Me If You Can: Human-in-the-Loop Generation of Adversarial Examples for Question Answering. Eric Wallace, Pedro Rodriguez, Shi Feng, Ikuya Yamada, Jordan Boyd-Graber. TACL 2019. score [pdf]
10.	PAWS: Paraphrase Adversaries from Word Scrambling. Yuan Zhang, Jason Baldridge, Luheng He. NAACL-HLT 2019. blind [pdf] [dataset]
11.	Evaluating and Enhancing the Robustness of Dialogue Systems: A Case Study on a Negotiation Agent. Minhao Cheng, Wei Wei, Cho-Jui Hsieh. NAACL-HLT 2019. gradient score [pdf] [code]
12.	Semantically Equivalent Adversarial Rules for Debugging NLP Models. Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. ACL 2018. decision [pdf] [code]
13.	Adversarially Regularising Neural NLI Models to Integrate Logical Background Knowledge. Pasquale Minervini, Sebastian Riedel. CoNLL 2018. score [pdf] [code&data]
14.	Robust Machine Comprehension Models via Adversarial Training. Yicheng Wang, Mohit Bansal. NAACL-HLT 2018. decision [pdf] [dataset]
15.	Adversarial Example Generation with Syntactically Controlled Paraphrase Networks. Mohit Iyyer, John Wieting, Kevin Gimpel, Luke Zettlemoyer. NAACL-HLT 2018. blind [pdf] [code&data]
16.	Generating Natural Adversarial Examples. Zhengli Zhao, Dheeru Dua, Sameer Singh. ICLR 2018. decision [pdf] [code]
17.	Adversarial Examples for Evaluating Reading Comprehension Systems. Robin Jia, Percy Liang. EMNLP 2017. score decision blind [pdf] [code]
18.	Adversarial Sets for Regularising Neural Link Predictors. Pasquale Minervini, Thomas Demeester, Tim Rocktäschel, Sebastian Riedel. UAI 2017. score [pdf] [code]

2.2 Word-level Attack

1.	Bridge the Gap Between CV and NLP! A Gradient-based Textual Adversarial Attack Framework. Lifan Yuan, Yichi Zhang, Yangyi Chen, Wei Wei. Findings of ACL 2023. decision[pdf]
2.	TextHacker: Learning based Hybrid Local Search Algorithm for Text Hard-label Adversarial Attack. Zhen Yu, Xiaosen Wang, Wanxiang Che, Kun He. Findings of EMNLP 2022. decision[pdf][code]
3.	TextHoaxer: Budgeted Hard-Label Adversarial Attacks on Text. Muchao Ye, Chenglin Miao, Ting Wang, Fenglong Ma. AAAI 2022. decision [pdf] [code]
4.	Query-Efficient and Scalable Black-Box Adversarial Attacks on Discrete Sequential Data via Bayesian Optimization. Deokjae Lee, Seungyong Moon, Junhyeok Lee, Hyun Oh Song. ICML 2022. score [pdf][code]
5.	SemAttack: Natural Textual Attacks on Different Semantic Spaces. Boxin Wang, Chejian Xu, Xiangyu Liu, Yu Cheng, Bo Li. Findings of NAACL 2022. gradient [pdf] [code]
6.	Gradient-based Adversarial Attacks against Text Transformers. Chuan Guo, Alexandre Sablayrolles, Hervé Jégou, Douwe Kiela. EMNLP 2021. gradient [pdf] [code]
7.	A Strong Baseline for Query Efficient Attacks in a Black Box Setting. Rishabh Maheswary, Saket Maheshwary, Vikram Pudi. EMNLP 2021. score [pdf] [code]
8.	On the Transferability of Adversarial Attacks against Neural Text Classifier. Liping Yuan, Xiaoqing Zheng, Yi Zhou, Cho-Jui Hsieh, Kai-Wei Chang. EMNLP 2021. [pdf]
9.	Crafting Adversarial Examples for Neural Machine Translation. Xinze Zhang, Junzhe Zhang, Zhenhua Chen, Kun He. ACL-IJCNLP 2021. score [pdf] [code]
10.	An Empirical Study on Adversarial Attack on NMT: Languages and Positions Matter. Zhiyuan Zeng, Deyi Xiong. ACL-IJCNLP 2021. score [pdf]
11.	A Closer Look into the Robustness of Neural Dependency Parsers Using Better Adversarial Examples. Yuxuan Wang, Wanxiang Che, Ivan Titov, Shay B. Cohen, Zhilin Lei, Ting Liu. Findings of ACL: ACL-IJCNLP 2021. score [pdf] [code]
12.	Contextualized Perturbation for Textual Adversarial Attack. Dianqi Li, Yizhe Zhang, Hao Peng, Liqun Chen, Chris Brockett, Ming-Ting Sun, Bill Dolan. NAACL-HLT 2021. score [pdf] [code]
13.	Adv-OLM: Generating Textual Adversaries via OLM. Vijit Malik, Ashwani Bhat, Ashutosh Modi. EACL 2021. score [pdf] [code]
14.	Adversarial Stylometry in the Wild: Transferable Lexical Substitution Attacks on Author Proﬁling. Chris Emmery, Ákos Kádár, Grzegorz Chrupała. EACL 2021. blind [pdf] [code]
15.	Generating Natural Language Attacks in a Hard Label Black Box Setting. Rishabh Maheshwary, Saket Maheshwary, Vikram Pudi. AAAI 2021. decision [pdf] [code]
16.	A Geometry-Inspired Attack for Generating Natural Language Adversarial Examples. Zhao Meng, Roger Wattenhofer. COLING 2020. gradient [pdf] [code]
17.	BERT-ATTACK: Adversarial Attack Against BERT Using BERT. Linyang Li, Ruotian Ma, Qipeng Guo, Xiangyang Xue, Xipeng Qiu. EMNLP 2020. score [pdf] [code]
18.	BAE: BERT-based Adversarial Examples for Text Classification. Siddhant Garg, Goutham Ramakrishnan. EMNLP 2020. score [pdf] [code]
19.	Detecting Word Sense Disambiguation Biases in Machine Translation for Model-Agnostic Adversarial Attacks. Denis Emelin, Ivan Titov, Rico Sennrich. EMNLP 2020. blind [pdf] [code]
20.	Imitation Attacks and Defenses for Black-box Machine Translation Systems. Eric Wallace, Mitchell Stern, Dawn Song. EMNLP 2020. decision [pdf] [code]
21.	Robustness to Modification with Shared Words in Paraphrase Identification. Zhouxing Shi, Minlie Huang. Findings of ACL: EMNLP 2020. score [pdf]
22.	Word-level Textual Adversarial Attacking as Combinatorial Optimization. Yuan Zang, Fanchao Qi, Chenghao Yang, Zhiyuan Liu, Meng Zhang, Qun Liu, Maosong Sun. ACL 2020. score [pdf] [code]
23.	It's Morphin' Time! Combating Linguistic Discrimination with Inflectional Perturbations. Samson Tan, Shafiq Joty, Min-Yen Kan, Richard Socher. ACL 2020. score [pdf] [code]
24.	On the Robustness of Language Encoders against Grammatical Errors. Fan Yin, Quanyu Long, Tao Meng, Kai-Wei Chang. ACL 2020. score [pdf] [code]
25.	Evaluating and Enhancing the Robustness of Neural Network-based Dependency Parsing Models with Adversarial Examples. Xiaoqing Zheng, Jiehang Zeng, Yi Zhou, Cho-Jui Hsieh, Minhao Cheng, Xuanjing Huang. ACL 2020. gradient score [pdf] [code]
26.	A Reinforced Generation of Adversarial Examples for Neural Machine Translation. Wei Zou, Shujian Huang, Jun Xie, Xinyu Dai, Jiajun Chen. ACL 2020. decision [pdf]
27.	Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment. Di Jin, Zhijing Jin, Joey Tianyi Zhou, Peter Szolovits. AAAI 2020. score [pdf] [code]
28.	Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples. Minhao Cheng, Jinfeng Yi, Pin-Yu Chen, Huan Zhang, Cho-Jui Hsieh. AAAI 2020. score [pdf] [code]
29.	Greedy Attack and Gumbel Attack: Generating Adversarial Examples for Discrete Data. Puyudi Yang, Jianbo Chen, Cho-Jui Hsieh, Jane-LingWang, Michael I. Jordan. JMLR 2020. score [pdf] [code]
30.	On the Robustness of Self-Attentive Models. Yu-Lun Hsieh, Minhao Cheng, Da-Cheng Juan, Wei Wei, Wen-Lian Hsu, Cho-Jui Hsieh. ACL 2019. score [pdf]
31.	Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency. Shuhuai Ren, Yihe Deng, Kun He, Wanxiang Che. ACL 2019. score [pdf] [code]
32.	Generating Fluent Adversarial Examples for Natural Languages. Huangzhao Zhang, Hao Zhou, Ning Miao, Lei Li. ACL 2019. gradient score [pdf] [code]
33.	Robust Neural Machine Translation with Doubly Adversarial Inputs. Yong Cheng, Lu Jiang, Wolfgang Macherey. ACL 2019. gradient [pdf]
34.	Universal Adversarial Attacks on Text Classifiers. Melika Behjati, Seyed-Mohsen Moosavi-Dezfooli, Mahdieh Soleymani Baghshah, Pascal Frossard. ICASSP 2019. gradient [pdf]
35.	Generating Natural Language Adversarial Examples. Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, Kai-Wei Chang. EMNLP 2018. score [pdf] [code]
36.	Breaking NLI Systems with Sentences that Require Simple Lexical Inferences. Max Glockner, Vered Shwartz, Yoav Goldberg. ACL 2018. blind [pdf] [dataset]
37.	Deep Text Classification Can be Fooled. Bin Liang, Hongcheng Li, Miaoqiang Su, Pan Bian, Xirong Li, Wenchang Shi. IJCAI 2018. gradient score [pdf]
38.	Interpretable Adversarial Perturbation in Input Embedding Space for Text. Sato, Motoki, Jun Suzuki, Hiroyuki Shindo, Yuji Matsumoto. IJCAI 2018. gradient [pdf] [code]
39.	Towards Crafting Text Adversarial Samples. Suranjana Samanta, Sameep Mehta. ECIR 2018. gradient [pdf]
40.	Crafting Adversarial Input Sequences For Recurrent Neural Networks. Nicolas Papernot, Patrick McDaniel, Ananthram Swami, Richard Harang. MILCOM 2016. gradient [pdf]

2.3 Char-level Attack

1.	Using Punctuation as an Adversarial Attack on Deep Learning-Based NLP Systems: An Empirical Study. Brian Formento, Chuan Sheng Foo, Luu Anh Tuan, See Kiong Ng. EACL (Findings) 2023. score blind [pdf] [code]
2.	Model Extraction and Adversarial Transferability, Your BERT is Vulnerable!. Xuanli He, Lingjuan Lyu, Lichao Sun, Qiongkai Xu. NAACL-HLT 2021. blind [pdf] [code]
3.	Text Processing Like Humans Do: Visually Attacking and Shielding NLP Systems. Steffen Eger, Gözde Gül ¸Sahin, Andreas Rücklé, Ji-Ung Lee, Claudia Schulz, Mohsen Mesgar, Krishnkant Swarnkar, Edwin Simpson, Iryna Gurevych. NAACL-HLT 2019. blind [pdf] [code&data]
4.	White-to-Black: Efficient Distillation of Black-Box Adversarial Attacks. SYotam Gil, Yoav Chai, Or Gorodissky, Jonathan Berant. NAACL-HLT 2019. blind [pdf] [code]
5.	Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers. Ji Gao, Jack Lanchantin, Mary Lou Soffa, Yanjun Qi. IEEE SPW 2018. score[pdf] [code]
6.	On Adversarial Examples for Character-Level Neural Machine Translation. Javid Ebrahimi, Daniel Lowd, Dejing Dou. COLING 2018. gradient [pdf] [code]
7.	Synthetic and Natural Noise Both Break Neural Machine Translation. Yonatan Belinkov, Yonatan Bisk. ICLR 2018. blind [pdf] [code&data]

2.4 Multi-level Attack

1.	Multi-granularity Textual Adversarial Attack with Behavior Cloning. Yangyi Chen, Jin Su, Wei Wei. EMNLP 2021. blind [pdf] [code]
2.	Synthesizing Adversarial Negative Responses for Robust Response Ranking and Evaluation. Prakhar Gupta, Yulia Tsvetkov, Jeffrey Bigham. Findings of ACL: ACL-IJCNLP 2021. blind [pdf] [code]
3.	Code-Mixing on Sesame Street: Dawn of the Adversarial Polyglots. Samson Tan, Shafiq Joty. NAACL-HLT 2021. score [pdf] [code]
4.	Universal Adversarial Attacks with Natural Triggers for Text Classification. Liwei Song, Xinwei Yu, Hsuan-Tung Peng, Karthik Narasimhan. NAACL-HLT 2021. gradient [pdf] [code]
5.	BBAEG: Towards BERT-based Biomedical Adversarial Example Generation for Text Classification. Ishani Mondal. NAACL-HLT 2021. score [pdf] [code]
6.	Don’t take “nswvtnvakgxpm” for an answer –The surprising vulnerability of automatic content scoring systems to adversarial input. Yuning Ding, Brian Riordan, Andrea Horbach, Aoife Cahill, Torsten Zesch. COLING 2020. blind [pdf] [code]
7.	Universal Adversarial Triggers for Attacking and Analyzing NLP. Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, Sameer Singh. EMNLP-IJCNLP 2019. gradient [pdf] [code] [website]
8.	TEXTBUGGER: Generating Adversarial Text Against Real-world Applications. Jinfeng Li, Shouling Ji, Tianyu Du, Bo Li, Ting Wang. NDSS 2019. gradient score [pdf]
9.	Generating Black-Box Adversarial Examples for Text Classifiers Using a Deep Reinforced Model. Prashanth Vijayaraghavan, Deb Roy. ECMLPKDD 2019. score [pdf]
10.	HotFlip: White-Box Adversarial Examples for Text Classification. Javid Ebrahimi, Anyi Rao, Daniel Lowd, Dejing Dou. ACL 2018. gradient [pdf] [code]
11.	Adversarial Over-Sensitivity and Over-Stability Strategies for Dialogue Models. Tong Niu, Mohit Bansal. CoNLL 2018. blind [pdf] [code&data]
12.	Comparing Attention-based Convolutional and Recurrent Neural Networks: Success and Limitations in Machine Reading Comprehension. Matthias Blohm, Glorianna Jagfeld, Ekta Sood, Xiang Yu, Ngoc Thang Vu. CoNLL 2018. gradient [pdf] [code]

3. Defense Papers

1.	Textual Manifold-based Defense against Natural Language Adversarial Examples. Dang Minh Nguyen, Luu Anh Tuan. EMNLP 2022. [pdf] [code]
2.	Detecting Word-Level Adversarial Text Attacks via SHapley Additive exPlanations. Lukas Huber, Marc Alexander Kühn, Edoardo Mosca, Georg Groh. Repl4NLP@ACL 2022. [pdf] [code]
3.	Detection of Adversarial Examples in Text Classification: Benchmark and Baseline via Robust Density Estimation. KiYoon Yoo, Jangho Kim, Jiho Jang, Nojun Kwawk. ACL 2022 (Findings). [pdf] [code]
4.	“That Is a Suspicious Reaction!”: Interpreting Logits Variation to Detect NLP Adversarial Attacks. Edoardo Mosca, Shreyash Agarwal, Javier Rando Ramírez, Georg Groh. ACL 2022. [pdf] [code]
5.	SHIELD: Defending Textual Neural Networks against Multiple Black-Box Adversarial Attacks with Stochastic Multi-Expert Patcher. Thai Le, Noseong Park, Dongwon Lee. ACL 2022. [pdf]
6.	Perturbations in the Wild: Leveraging Human-Written Text Perturbations for Realistic Adversarial Attack and Defense. Thai Le, Jooyoung Lee, Kevin Yen, Yifan Hu, Dongwon Lee. ACL 2022 (Findings). [pdf]
7.	Achieving Model Robustness through Discrete Adversarial Training. Maor Ivgi, Jonathan Berant. EMNLP 2021. [pdf] [code]
8.	Defense against Synonym Substitution-based Adversarial Attacks via Dirichlet Neighborhood Ensemble. Yi Zhou, Xiaoqing Zheng, Cho-Jui Hsieh, Kai-Wei Chang, Xuanjing Huang. ACL-IJCNLP 2021. [pdf]
9.	A Sweet Rabbit Hole by DARCY: Using Honeypots to Detect Universal Trigger’s Adversarial Attacks. Thai Le, Noseong Park, Dongwon Lee. ACL-IJCNLP 2021. [pdf] [code]
10.	Better Robustness by More Coverage: Adversarial and Mixup Data Augmentation for Robust Finetuning. Chenglei Si, Zhengyan Zhang, Fanchao Qi, Zhiyuan Liu, Yasheng Wang, Qun Liu, Maosong Sun. Findings of ACL: ACL-IJCNLP 2021. [pdf] [code]
11.	BERT-Defense: A Probabilistic Model Based on BERT to Combat Cognitively Inspired Orthographic Adversarial Attacks. Yannik Keller, Jan Mackensen, Steffen Eger. Findings of ACL: ACL-IJCNLP 2021. [pdf] [code]
12.	Defending Pre-trained Language Models from Adversarial Word Substitution Without Performance Sacrifice. Rongzhou Bao, Jiayi Wang, Hai Zhao. Findings of ACL: ACL-IJCNLP 2021. [pdf] [code]
13.	Manifold Adversarial Augmentation for Neural Machine Translation. Guandan Chen, Kai Fan, Kaibo Zhang, Boxing Chen, Zhongqiang Huang. Findings of ACL: ACL-IJCNLP 2021. [pdf]
14.	Natural Language Adversarial Defense through Synonym Encoding. Xiaosen Wang, Hao Jin, Kun He. UAI 2021. [pdf] [code]
15.	Adversarial Training with Fast Gradient Projection Method against Synonym Substitution based Text Attacks. Xiaosen Wang, Yichen Yang, Yihe Deng, Kun He. AAAI 2021. [pdf] [code]
16.	Frequency-Guided Word Substitutions for Detecting Textual Adversarial Examples. Maximilian Mozes, Pontus Stenetorp, Bennett Kleinberg, Lewis D. Griffin. EACL 2021. [pdf] [code]
17.	Towards Robustness Against Natural Language Word Substitutions. Xinshuai Dong, Anh Tuan Luu, Rongrong Ji, Hong Liu. ICLR 2021. [pdf] [code]
18.	InfoBERT: Improving Robustness of Language Models from An Information Theoretic Perspective. Boxin Wang, Shuohang Wang, Yu Cheng, Zhe Gan, Ruoxi Jia, Bo Li, Jingjing Liu. ICLR 2021. [pdf] [code]
19.	Enhancing Neural Models with Vulnerability via Adversarial Attack. Rong Zhang, Qifei Zhou, Bo An, Weiping Li, Tong Mo, Bo Wu. COLING 2020. [pdf] [code]
20.	Contrastive Zero-Shot Learning for Cross-Domain Slot Filling withAdversarial Attack. Keqing He, Jinchao Zhang, Yuanmeng Yan, Weiran Xu, Cheng Niu, Jie Zhou. COLING 2020. [pdf]
21.	Mind Your Inflections! Improving NLP for Non-Standard Englishes with Base-Inflection Encoding. Samson Tan, Shafiq Joty, Lav R. Varshney, Min-Yen Kan. EMNLP 2020. [pdf] [code]
22.	Robust Encodings: A Framework for Combating Adversarial Typos. Erik Jones, Robin Jia, Aditi Raghunathan, Percy Liang. ACL 2020. [pdf] [code]
23.	Joint Character-level Word Embedding and Adversarial Stability Training to Defend Adversarial Text. Hui Liu, Yongzheng Zhang, Yipeng Wang, Zheng Lin, Yige Chen. AAAI 2020. [pdf]
24.	A Robust Adversarial Training Approach to Machine Reading Comprehension. Kai Liu, Xin Liu, An Yang, Jing Liu, Jinsong Su, Sujian Li, Qiaoqiao She. AAAI 2020. [pdf]
25.	FreeLB: Enhanced Adversarial Training for Language Understanding. Chen Zhu, Yu Cheng, Zhe Gan, Siqi Sun, Tom Goldstein, Jingjing Liu. CoRR 2019. [pdf] [code]
26.	Learning to Discriminate Perturbations for Blocking Adversarial Attacks in Text Classification. Yichao Zhou, Jyun-Yu Jiang, Kai-Wei Chang, Wei Wang. EMNLP-IJCNLP 2019. [pdf] [code]
27.	Build it Break it Fix it for Dialogue Safety: Robustness from Adversarial Human Attack. Emily Dinan, Samuel Humeau, Bharath Chintagunta, Jason Weston. EMNLP-IJCNLP 2019. [pdf] [data]
28.	Combating Adversarial Misspellings with Robust Word Recognition. Danish Pruthi, Bhuwan Dhingra, Zachary C. Lipton. ACL 2019. [pdf] [code]
29.	Robust-to-Noise Models in Natural Language Processing Tasks. Valentin Malykh. ACL 2019. [pdf] [code]

4. Certified Robustness

1.	Certified Robustness to Word Substitution Attack with Differential Privacy. Wenjie Wang, Pengfei Tang, Jian Lou, Li Xiong. NAACL-HLT 2021. [pdf]
2.	Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond. Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Kai-Wei Chang, Minlie Huang, Bhavya Kailkhura, Xue Lin, Cho-Jui Hsieh. NeurIPS 2020. [pdf] [code]
3.	SAFER: A Structure-free Approach for Certified Robustness to Adversarial Word Substitutions. Mao Ye, Chengyue Gong, Qiang Liu. ACL 2020. [pdf] [code]
4.	Robustness Verification for Transformers. Zhouxing Shi, Huan Zhang, Kai-Wei Chang, Minlie Huang, Cho-Jui Hsieh. ICLR 2020. [pdf] [code]
5.	Achieving Verified Robustness to Symbol Substitutions via Interval Bound Propagation. Po-Sen Huang, Robert Stanforth, Johannes Welbl, Chris Dyer, Dani Yogatama, Sven Gowal, Krishnamurthy Dvijotham, Pushmeet Kohli. EMNLP-IJCNLP 2019. [pdf]
6.	Certified Robustness to Adversarial Word Substitutions. Robin Jia, Aditi Raghunathan, Kerem Göksel, Percy Liang. EMNLP-IJCNLP 2019. [pdf] [code]
7.	POPQORN: Quantifying Robustness of Recurrent Neural Networks. Ching-Yun Ko, Zhaoyang Lyu, Lily Weng, Luca Daniel, Ngai Wong, Dahua Lin. ICML 2019. [pdf] [code]
5. Benchmark and Evaluation
1.	Preserving Semantics in Textual Adversarial Attacks. David Herel, Hugo Cisneros, Tomas Mikolov. ECAI 2023. [pdf] [code]
2.	Prompting GPT-3 To Be Reliable. Chenglei Si, Zhe Gan, Zhengyuan Yang, Shuohang Wang, Jianfeng Wang, Jordan Boyd-Graber, Lijuan Wang. ICLR 2023. [pdf] [code]
3.	Why Should Adversarial Perturbations be Imperceptible? Rethink the Research Paradigm in Adversarial NLP. Yangyi Chen, Hongcheng Gao, Ganqu Cui, Fanchao Qi, Longtao Huang, Zhiyuan Liu, Maosong Sun. EMNLP 2022. [pdf] [code&data]
4.	Interpreting the Robustness of Neural NLP Models to Textual Perturbations. Yunxiang Zhang, Liangming Pan, Samson Tan, Min-Yen Kan. Findings of ACL, 2022. [pdf]
5.	Contrasting Human- and Machine-Generated Word-Level Adversarial Examples for Text Classification. Maximilian Mozes, Max Bartolo, Pontus Stenetorp, Bennett Kleinberg, Lewis D. Griffin. EMNLP 2021. [pdf] [code]
6.	Dynabench: Rethinking Benchmarking in NLP. Douwe Kiela, Max Bartolo, Yixin Nie, Divyansh Kaushik, Atticus Geiger, Zhengxuan Wu, Bertie Vidgen, Grusha Prasad, Amanpreet Singh, Pratik Ringshia, Zhiyi Ma, Tristan Thrush, Sebastian Riedel, Zeerak Waseem, Pontus Stenetorp, Robin Jia, Mohit Bansal, Christopher Potts, Adina Williams. NAACL 2021. [pdf] [website]
7.	Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models. Boxin Wang, Chejian Xu, Shuohang Wang, Zhe Gan, Yu Cheng, Jianfeng Gao, Ahmed Hassan Awadallah, Bo Li. NeurIPS 2021 (Datasets and Benchmarks Track). [pdf] [website]
8.	Searching for an Effiective Defender: Benchmarking Defense against Adversarial Word Substitution. Zongyi Li, Jianhan Xu, Jiehang Zeng, Linyang Li, Xiaoqing Zheng, Qi Zhang, Kai-Wei Chang, and Cho-Jui Hsieh. EMNLP 2021. [pdf]
9.	Double Perturbation: On the Robustness of Robustness and Counterfactual Bias Evaluation. Chong Zhang, Jieyu Zhao, Huan Zhang, Kai-Wei Chang, and Cho-Jui Hsieh NAACL 2021. [pdf] [code]
10.	Reevaluating Adversarial Examples in Natural Language. John Morris, Eli Lifland, Jack Lanchantin, Yangfeng Ji, Yanjun Qi. Findings of ACL: EMNLP 2020. [pdf] [code&data]
11.	From Hero to Zéroe: A Benchmark of Low-Level Adversarial Attacks. Steffen Eger, Yannik Benz. AACL-IJCNLP 2020. [pdf] [code&data]
12.	Adversarial NLI: A New Benchmark for Natural Language Understanding. Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, Douwe Kiela. ACL 2020. [pdf] [demo] [dataset & leaderboard]
13.	Evaluating NLP Models via Contrast Sets. Matt Gardner, Yoav Artzi, Victoria Basmova, Jonathan Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi, Dheeru Dua, Yanai Elazar, Ananth Gottumukkala, Nitish Gupta, Hanna Hajishirzi, Gabriel Ilharco, Daniel Khashabi, Kevin Lin, Jiangming Liu, Nelson F. Liu, Phoebe Mulcaire, Qiang Ning, Sameer Singh, Noah A. Smith, Sanjay Subramanian, Reut Tsarfaty, Eric Wallace, Ally Zhang, Ben Zhou. Findings of ACL: EMNLP 2020. [pdf] [website]
14.	On Evaluation of Adversarial Perturbations for Sequence-to-Sequence Models. Paul Michel, Xian Li, Graham Neubig, Juan Miguel Pino. NAACL-HLT 2019. [pdf] [code]

6. Other Papers

1.	Identifying Human Strategies for Generating Word-Level Adversarial Examples. Maximilian Mozes, Bennett Kleinberg, Lewis D. Griffin. Findings of ACL: EMNLP 2022. [pdf]
2.	LexicalAT: Lexical-Based Adversarial Reinforcement Training for Robust Sentiment Classification. Jingjing Xu, Liang Zhao, Hanqi Yan, Qi Zeng, Yun Liang, Xu Sun. EMNLP-IJCNLP 2019. [pdf] [code]
3.	Unified Visual-Semantic Embeddings: Bridging Vision and Language with Structured Meaning Representations. Hao Wu, Jiayuan Mao, Yufeng Zhang, Yuning Jiang, Lei Li, Weiwei Sun, Wei-Ying Ma. CVPR 2019. [pdf]
4.	AdvEntuRe: Adversarial Training for Textual Entailment with Knowledge-Guided Examples. Dongyeop Kang, Tushar Khot, Ashish Sabharwal, Eduard Hovy. ACL 2018. [pdf] [code]
5.	Learning Visually-Grounded Semantics from Contrastive Adversarial Samples. Haoyue Shi, Jiayuan Mao, Tete Xiao, Yuning Jiang, Jian Sun. COLING 2018. [pdf] [code]
