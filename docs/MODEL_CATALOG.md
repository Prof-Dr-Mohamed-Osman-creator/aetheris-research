# Aetheris Research — Complete Model Catalogue
# الدليل الشامل للنماذج والاختبارات

> **الهدف:** أن يجد أي باحث، في أي دولة وبأي مستوى (طالب مبتدئ، أو باحث يستهدف النشر الدولي، أو صانع قرار)،
> كل النماذج المعروفة في الإحصاء والاقتصاد القياسي والاقتصاد التطبيقي وبحوث العمليات والتنبؤ، في منصة واحدة.
>
> هذا الملف هو **المرجع الملزم للتطوير**. كل نموذج يُنفَّذ يُعلَّم بـ ✅.
> الحالة: ✅ منفّذ · 🔜 الإصدار القادم · ⬜ مخطط

---

## 0. المعيار الموحد لكل نموذج (Every model ships with all of this)

| # | العنصر | Element |
|---|---|---|
| 1 | معالج خطوة بخطوة للمبتدئ، ووضع مختصر للخبير | Guided wizard + expert mode |
| 2 | فحص الافتراضات قبل التقدير | Pre-estimation assumption checks |
| 3 | جدول تقدير بنمط المجلات (عدة نماذج متجاورة، نجوم المعنوية، الأخطاء المعيارية بين قوسين) | Journal-style multi-column estimation table |
| 4 | لوحة التشخيص الكامل بعد التقدير | Full post-estimation diagnostics |
| 5 | جدول المرونات أو الآثار الحدية مع فترات الثقة | Elasticities / marginal effects with CIs |
| 6 | رسوم بيانية خاصة بالنموذج (منحنيات، بواقي، تنبؤ، دوال استجابة) | Model-specific charts |
| 7 | قراءة أكاديمية تلقائية بالعربية والإنجليزية | Automatic bilingual interpretation |
| 8 | **ملخص لصانع القرار**: ماذا تعني النتيجة عملياً، ومحاكاة «ماذا لو» | Policy brief + what-if simulation |
| 9 | فقرة منهجية جاهزة مع المراجع العلمية للنموذج | Auto-generated methodology paragraph + references |
| 10 | تصدير Excel وWord وLaTeX، وكود Python / R / Stata لإعادة التحليل | Export + reproducible code |

---

## 1. محور الاقتصاد القياسي — Econometrics
> يضم 26 محوراً فرعياً. لكل نموذج: فحص الافتراضات، والتقدير، وجدول بنمط المجلات، والتشخيص، والرسوم الخاصة به، والتفسير، وملخص السياسات.
> 🧠 = نماذج الذكاء الاصطناعي والتعلم العميق التي تحتاج خادم حوسبة أقوى (انظر القسم 9).

### 1.1 نماذج الانحدار الكلاسيكية والمقطعية — Classical & Cross-Sectional Regression ✅ (v1.4)
- ✅ الانحدار الخطي البسيط (SLR) والمتعدد (MLR) · Simple & multiple linear regression
- ✅ الانحدار المتعدد الحدود، والتدريجي (أمامي، خلفي، أفضل مجموعة)، والهرمي · Polynomial, stepwise, best subsets, hierarchical
- ✅ الصور الدالية (خطي، لوغاريتمي مزدوج، نصف لوغاريتمي، مقلوب، أسي) وتحويل Box-Cox · Functional forms, Box-Cox
- ✅ نماذج المتغيرات الصورية وتحليل التغاير (ANCOVA)، والتفاعل والتأثير المعدِّل، والتأثير الوسيط · Dummy variables, ANCOVA, moderation, mediation
- ✅ المربعات الصغرى الموزونة والمعممة (WLS، GLS، FGLS)، والانحدار المتين، والمربعات الصغرى غير الخطية والمقيدة · WLS / GLS / FGLS, robust, NLS, restricted LS

> **v1.4 — منفّذ:** الصور الدالية الخمس مع Box-Cox، والمتغيرات الصورية وANCOVA مع المتوسطات المعدلة، والتفاعل (الميول البسيطة وجونسون-نيمان) والوساطة (سوبل والبوتستراب)، والتدريجي وأفضل مجموعة والهرمي، وOLS وWLS وFGLS وGLS-AR(1) والمتين (هوبر وتوكي) والمقيد (RLS) وغير الخطي (6 صيغ)، والأخطاء المعيارية HC0–HC3 ونيوي-ويست والعنقودية والبوتستراب، واختبارات والد وLR وLM؛ مع المعيار الموحد كاملاً (معالج موجّه، فحص قبلي، جداول المجلات، المرونات بطريقة دلتا، الرسوم، التفسير ثنائي اللغة، ملخص صانع القرار و«ماذا لو»، فقرة المنهجية والمراجع، تصدير Excel وWord وLaTeX وكود Python وR وStata).

### 1.2 نماذج المعادلات الآنية والمتغيرات الأداتية — Simultaneous Equations & IV 🔜 (v1.5)
- نماذج المعادلات الآنية (SEM) ومسألة التعرّف (شرطا الرتبة والدرجة) · Simultaneous equations, identification
- المربعات الصغرى غير المباشرة (ILS)، و**ذات المرحلتين (2SLS)**، و**ذات الثلاث مراحل (3SLS)**، وLIML، وGMM · ILS, 2SLS, 3SLS, LIML, GMM
- الانحدارات غير المرتبطة ظاهرياً (SUR) · SUR
- اختبارات ضعف الأدوات، وفرط التحديد (سارجان وهانسن)، والداخلية (دربن-وو-هاوسمان) · Weak-IV, over-identification, endogeneity tests

### 1.3 التشخيص الكامل — Full Diagnostics (لكل نماذج الانحدار) ✅ (v1.4)
- ✅ عدم تجانس التباين: بروش-باجان، وايت، جليجسر، بارك، جولدفيلد-كواندت
- ✅ الارتباط الذاتي: دربن-واتسون، بروش-جودفري، ليونج-بوكس، دربن h
- ✅ الازدواج الخطي: VIF، ودليل الحالة · الشكل الدالي والاستقرار: RESET، شاو، CUSUM وCUSUMSQ
- ✅ الطبيعية والمشاهدات المؤثرة: جارك-بيرا، مسافة كوك، الرفع، DFBETAS
- ✅ الأخطاء المعيارية المتينة: HC0–HC3، نيوي-ويست، العنقودية، البوتستراب · اختبارات والد وLR وLM، ومعايير AIC وBIC

### 1.4 نماذج الاستجابة النوعية والمتغير التابع المحدود — Limited Dependent Variable 🔜 (v1.5)
- نموذج الاحتمال الخطي (LPM) · Linear probability model
- اللوجيت والبروبيت وComplementary log-log، مع الآثار الحدية ونسب الأرجحية ومنحنى ROC · Logit, Probit, cloglog
- اللوجيت والبروبيت متعدد الحدود، واللوجيت والبروبيت المرتب · Multinomial & ordered logit / probit
- التوبيت، والانحدار المبتور، ونموذج هيكمان للاختيار، والعقبة المزدوجة، والاستجابة الكسرية · Tobit, truncated, Heckman, double hurdle, fractional response

### 1.5 نماذج بيانات العدّ والبقاء — Count & Duration / Survival 🔜 (v1.5)
- بواسون، وذو الحدين السالب، والتضخم الصفري (ZIP وZINB)، والعقبة، وPPML · Poisson, NegBin, ZIP / ZINB, hurdle, PPML
- نماذج البقاء والمدة: كابلان-ماير، والمخاطر النسبية لكوكس، وويبل ونماذج الزمن المتسارع، ونماذج المخاطر في الزمن المتقطع · Kaplan-Meier, Cox PH, Weibull / AFT, discrete-time hazard

### 1.6 نماذج الخيارات المتقطعة الهيكلية والمنفعة العشوائية — Structural Discrete Choice & IO
- اللوجيت الشرطي، والمتداخل (Nested)، والمختلط (Mixed / Random Parameters)، وفئات كامنة · Conditional, nested, mixed, latent-class logit
- نموذج بيري-ليفينسون-بايكس (BLP) للطلب والتنظيم الصناعي · BLP
- نماذج الخيار المتقطع الديناميكية (رست) والألعاب الديناميكية الهيكلية · Dynamic discrete choice (Rust), dynamic games
- تجارب الاختيار والرغبة في الدفع · Choice experiments, willingness to pay

### 1.7 نماذج السلاسل الزمنية الأحادية — Univariate Time Series
- الاختبارات التمهيدية: التفكيك، والمرشحات (HP، BK، CF)، و**جذور الوحدة** (ADF، PP، KPSS، DF-GLS، Ng-Perron، HEGY الموسمي)
- الانحدار الذاتي (AR)، والمتوسطات المتحركة (MA)، وARMA · AR, MA, ARMA
- **ARIMA (منهجية بوكس-جنكنز كاملة: التعرف، والتقدير، والتشخيص، والتنبؤ)** · ARIMA (Box-Jenkins)
- SARIMA الموسمية، وARFIMA للذاكرة الطويلة · SARIMA, ARFIMA

### 1.8 نماذج السلاسل الزمنية المتعددة والتكامل المشترك — Multivariate & Cointegration
- الانحدار الذاتي المتجهي (VAR) والهيكلي (SVAR): اختيار الإبطاء، والاستقرار، ودوال الاستجابة الفورية، وتجزئة التباين · VAR, SVAR, IRF, FEVD
- السببية لجرانجر وتودا-ياماموتو · Granger, Toda-Yamamoto causality
- التكامل المشترك: إنجل-جرانجر، وجوهانسن، وجريجوري-هانسن (مع الانكسار)، وFMOLS وDOLS وCCR · Engle-Granger, Johansen, Gregory-Hansen, FMOLS / DOLS / CCR
- نماذج تصحيح الخطأ (ECM) وتصحيح الخطأ المتجهي (VECM) · ECM, VECM
- **ARDL واختبار الحدود**، و**NARDL غير الخطي** (الآثار غير المتماثلة ومضاعفاتها الديناميكية)، وARDL الكمي (QARDL) · ARDL bounds, NARDL, QARDL

### 1.9 نماذج التقلب المالي — Volatility Models
- ARCH، وGARCH، وGARCH-M، وEGARCH، وTGARCH / GJR-GARCH · Univariate GARCH family
- متعددة المتغيرات: DCC-GARCH، وCCC، وBEKK · Multivariate GARCH

### 1.10 نماذج البيانات عالية التردد والمختلطة — High-Frequency & Mixed Data
- أخذ العينات بالبيانات المختلطة (MIDAS) · MIDAS
- التقلب المحقق (HAR-RV) · Realized volatility
- المدة الشرطية الانحدارية الذاتية (ACD) · ACD

### 1.11 نماذج العوامل والأبعاد العالية — Factor & High-Dimensional
- العوامل الديناميكية (DFM)، وVAR المعزز بالعوامل (FAVAR) · DFM, FAVAR
- لاسو، وريدج، والشبكة المرنة، والاختيار المزدوج بلاسو (Post-double-selection) · LASSO, Ridge, Elastic Net, PDS-Lasso
- المكونات الرئيسية والمربعات الصغرى الجزئية، وإكمال المصفوفات · PCR, PLS, matrix completion

### 1.12 نماذج المعاملات المتغيرة والهيكلية وغير الخطية — Time-Varying, Structural & Nonlinear
- VAR بمعاملات متغيرة زمنياً (TVP-VAR) · TVP-VAR
- الانحدار الانتقالي السلس (STR، LSTAR، ESTAR)، وعتبة الانحدار الذاتي (TAR، SETAR)، و**تغيير النظام الماركوفي** · STR / LSTAR / ESTAR, TAR / SETAR, Markov switching
- الإسقاطات المحلية (Jordà) · Local projections
- مؤشر الانتشار والترابط لديبولد-يلماز · Diebold-Yilmaz spillover index
- الانكسارات الهيكلية: شاو، وباي-بيرون للانكسارات المتعددة، وزيفوت-أندروز · Chow, Bai-Perron, Zivot-Andrews

### 1.13 النماذج اللامعلمية وشبه المعلمية — Nonparametric & Semi-parametric
- انحدار النواة، وكثيرات الحدود المحلية، والشرائح · Kernel, local polynomial, splines
- النماذج المضافة المعممة (GAM)، والنماذج الخطية الجزئية · GAM, partially linear models
- **الكوانتيل على الكوانتيل (QQR)** · Quantile-on-Quantile regression
- **تماسك المويجات** والمويجات الكوانتيلية · Wavelet coherence, wavelet quantile

### 1.14 الانحدار الكمي — Quantile Regression
- الانحدار الكمي (QR) مع رسم المعاملات عبر الكميات · Quantile regression
- البانل الكوانتيلي بالآثار الثابتة (MMQR) · Panel quantile (MMQR)
- مخطط الارتباط الكمي المتقاطع (Cross-quantilogram) · Cross-quantilogram

### 1.15 نماذج البيانات الطولية (البانل) — Panel Data
- **الساكنة:** التجميعي، والآثار الثابتة (أحادية وثنائية الاتجاه)، والآثار العشوائية، والبيني، والفروق الأولى · Pooled, FE, RE, between, FD
- **الاختبارات:** هاوسمان، وبروش-باجان LM، وF، والاعتماد المقطعي لبيساران (CD)، وتجانس الميل (بيساران-ياماجاتا)، وولدريدج
- **الأخطاء المعيارية:** دريسكول-كراي، وPCSE، والعنقودية، وFGLS
- **جذور الوحدة والتكامل:** LLC، وIPS، وFisher، وCIPS؛ بيدروني، وكاو، وويسترلوند
- **الديناميكية:** أريلانو-بوند، وبلونديل-بوند (System GMM) · Dynamic panel GMM
- **ARDL للبانل:** PMG، وMG، وDFE · Panel ARDL
- **المتطورة:** الآثار التفاعلية الثابتة (Bai)، والآثار المرتبطة الشائعة (CCE)، وعتبة البانل (هانسن)، وVAR للبانل · Interactive FE, CCE, panel threshold, panel VAR
- **السببية:** دوميترسكو-هورلن · Dumitrescu-Hurlin

### 1.16 الاقتصاد القياسي المكاني — Spatial Econometrics
- موران I، وLISA، والخرائط · Moran's I, LISA
- SAR، وSEM، وSDM، وSAC، والانحدار الموزون جغرافياً (GWR) · Spatial lag, error, Durbin, SAC, GWR
- البانل المكاني · Spatial panel models

### 1.17 نماذج الشبكات والعدوى المالية — Network & Contagion
- القيمة المعرضة للمخاطر المشروطة (CoVaR)، والعجز المتوقع الهامشي (MES)، وSRISK · CoVaR, MES, SRISK
- نماذج الكوبولا · Copula models
- الشبكات المالية والاجتماعية · Financial & social network models

### 1.18 الاقتصاد القياسي البايزي — Bayesian Econometrics
- الانحدار البايزي ومحاكاة MCMC · Bayesian regression, MCMC
- **BVAR** (أولوية مينيسوتا) · BVAR
- متوسط النماذج البايزي (BMA) · Bayesian model averaging

### 1.19 النماذج الكلية الهيكلية والتوزيعية — Macro Structural & Distributional
- **التوازن العام الديناميكي العشوائي (DSGE)**: الحل والمعايرة والتقدير البايزي 🧠
- نماذج الوكلاء غير المتجانسين نيو-كينزية (HANK) 🧠
- الانحدار التوزيعي · Distributional regression
- التقدير بالمحاكاة (SMM)، والاستدلال غير المباشر · Simulated method of moments, indirect inference

### 1.20 تقييم السياسات والاستدلال السببي — Policy Evaluation & Causal Inference
- التجارب العشوائية (RCT) · Randomized trials
- **الفروق في الفروق (DiD)**، و**المتدرجة** (Callaway-Sant'Anna، Sun-Abraham)، ودراسة الحدث · DiD, staggered DiD, event study
- **الفروق في الفروق الاصطناعية (SDiD)**، و**الضبط الاصطناعي (SCM)** · Synthetic DiD, synthetic control
- **الانحدار المتقطع (RDD)** الحاد والغامض (Fuzzy) · Sharp & fuzzy RDD
- **مطابقة درجات الميل (PSM)**، والترجيح العكسي (IPW)، والمزدوج المتانة · PSM, IPW, doubly robust
- المتغيرات الأداتية والأثر المحلي (LATE) · IV / LATE

### 1.21 التعلم الآلي السببي — Causal Machine Learning
- **التعلم الآلي المزدوج / غير المتحيز (DML)** · Double/Debiased ML
- الغابات السببية والغابات العشوائية المعممة (GRF)، وشجرة القرار السببية (Athey-Imbens) · Causal forests, GRF, causal trees
- المتعلمون الفوقيون: S وT وX وR · Meta-learners
- التعلم المستهدف بالحد الأقصى للاحتمالية (TMLE) · TMLE
- الأشجار البايزية المضافة (BART) · BART

### 1.22 الاقتصاد القياسي العميق — Deep Econometrics 🧠
- DeepIV، وDeepGMM · Deep instrumental variables
- الشبكات التوليدية التنافسية السببية والمضادة للواقع (Causal / Counterfactual GANs) · Causal GANs
- التشفير التلقائي السببي (CEVAE) · CEVAE
- المحولات السببية · Causal Transformers

### 1.23 التنبؤ بالسلاسل الزمنية بالذكاء الاصطناعي — AI Time-Series Econometrics 🧠
- الشبكات الهجينة (ARIMA-ANN، GARCH-LSTM) · Hybrid models
- LSTM وGRU القياسية · Econometric LSTM / GRU
- N-BEATS وN-HiTS، والمحولات الزمنية (TFT) · N-BEATS / N-HiTS, Temporal Fusion Transformers
- التقلب العميق (Neural GARCH) · Deep volatility
- الشبكات العصبية البيانية للترابط الاقتصادي (GNN) · Graph neural networks

### 1.24 البيانات الضخمة والنصوص والصور — Unstructured & Big Data Econometrics
- **التنبؤ الآني (Nowcasting)** · Nowcasting
- النصوص كبيانات: تحليل المشاعر، ونمذجة الموضوعات (LDA) · Text-as-data, sentiment, LDA
- التضمين اللغوي (Word Embeddings، BERT) 🧠 · Econometric embeddings
- الاستدلال السببي بمساعدة النماذج اللغوية الكبيرة (LLM) · LLM-assisted causal inference
- الاقتصاد القياسي بصور الأقمار الصناعية والرؤية الحاسوبية (CNN لخرائط الفقر والناتج) 🧠 · Satellite & computer-vision econometrics

### 1.25 تحسين السياسات بالذكاء الاصطناعي — AI Policy Optimization 🧠
- التعلم المعزز للسياسات الديناميكية، وأنظمة العلاج الديناميكية (DTR) · Reinforcement learning, dynamic treatment regimes
- قطاع الطرق متعدد الأذرع السياقي للتجارب السياسية · Contextual multi-armed bandits
- المحاكاة الاقتصادية القائمة على الوكلاء الأذكياء (AI-ABM) · AI-driven agent-based models

### 1.26 اقتصاديات المناخ والطاقة — Climate & Energy Econometrics
- **نماذج استجابة الإنتاجية لدرجات الحرارة للبانل** (غير الخطية، ودرجات الحرارة النامية والضارة) · Temperature-response panel models, degree days
- النهج الريكاردي · Ricardian approach
- نماذج التقييم المتكامل القياسية (IAMs) · Econometric IAMs
- STIRPAT والتنبؤ بالانبعاثات والذروة الكربونية · STIRPAT, emissions peaking
- منحنى كوزنتس البيئي، ومؤشر فك الارتباط (تابيو)، وتجزئة LMDI · EKC, Tapio decoupling, LMDI decomposition

---

## 2. محور النماذج الاقتصادية التطبيقية — Applied Economic Models
> لكل نموذج: التقدير أو المعايرة + المرونات والمضاعفات + الرسوم الاقتصادية الخاصة به + التفسير الاقتصادي + **محاكاة السياسات لصانع القرار**.
> ⚙️ = نماذج هيكلية كبيرة تحتاج محركات حل رياضية وبيانات خاصة (انظر القسم 9) · 🧠 = نماذج الذكاء الاصطناعي.

### أولاً: النماذج الكلية والهيكلية — Macro & Structural

#### 2.1 نماذج التوازن العام والمحاكاة الكلية — General Equilibrium & Macro Simulation
- التوازن العام القابل للحساب (CGE) ⚙️، والديناميكي (DCGE) ⚙️ · Computable general equilibrium, dynamic CGE
- نماذج التوازن الجزئي (PE) لأسواق السلع والسياسات التجارية والزراعية · Partial equilibrium models
- التوازن العام الديناميكي العشوائي (DSGE)، والوكلاء غير المتجانسين النيو-كينزية (HANK) — انظر 1.19 🧠
- نماذج الأجيال المتداخلة (OLG) ⚙️ · Overlapping generations

#### 2.2 نماذج المدخلات والمخرجات والمحاسبة الاجتماعية — Input-Output & SAM
- نموذج ليونتيف: المعكوس، والمضاعفات (الإنتاج، والدخل، والتوظيف)، والروابط الأمامية والخلفية، والقطاعات الرائدة · Leontief IO, multipliers, linkages, key sectors
- المدخلات والمخرجات متعددة الأقاليم (MRIO)، والبصمة المائية والكربونية · Multi-regional IO, water & carbon footprints
- مصفوفة المحاسبة الاجتماعية (SAM) ونماذج مضاعفاتها، وتحليل المسار البنائي · SAM, SAM multipliers, structural path analysis

#### 2.3 نماذج النمو والتنمية الاقتصادية — Growth & Development
- سولو-سوان، ونماذج النمو الداخلي (رومر، ولوكاس)، وهارود-دومار · Solow-Swan, endogenous growth, Harrod-Domar
- نماذج الفجوة المزدوجة والثلاثية · Two-gap & three-gap models
- التحول الهيكلي (لويس، وتشينري) · Structural change (Lewis, Chenery)
- محاسبة النمو والإنتاجية الكلية (بواقي سولو)، والتقارب (بيتا وسيجما) · Growth accounting, TFP, convergence

#### 2.4 نماذج التجارة الدولية والاقتصاد المفتوح — International Trade & Open Economy
- نموذج الجاذبية للتجارة (PPML والآثار الثابتة المتعددة) · Gravity models
- هيكشر-أولين التطبيقي، وماندل-فليمنج، ونموذج دورنبوش لتجاوز سعر الصرف · Heckscher-Ohlin, Mundell-Fleming, Dornbusch overshooting
- مشروع تحليل التجارة العالمية (GTAP) ⚙️ · GTAP
- الميزة النسبية الظاهرة (RCA، وRSCA، ولافاي)، وتحليل حصة السوق الثابتة · Revealed comparative advantage, constant market share
- مؤشرات التركز وعدم الاستقرار للصادرات، والتجارة البينية (جروبل-لويد) · Export concentration & instability, intra-industry trade
- دالة الطلب على الواردات، وانتقال أثر سعر الصرف، وشرط مارشال-ليرنر، ومنحنى J · Import demand, ERPT, Marshall-Lerner, J-curve

#### 2.5 الاقتصاد الكلي الهيكلي والسياسات المالية والنقدية — Structural Macro & Policy
- النماذج الكلية القياسية واسعة النطاق (على غرار FRB/US) ⚙️ · Large-scale macroeconometric models
- منحنى فيليبس النيو-كينزي (NKPC)، وقاعدة تايلور · New Keynesian Phillips curve, Taylor rule
- **تحليل استدامة الدين العام (DSA)** مع رسم المروحة للدين · Debt sustainability analysis
- نماذج التوازن بين المخزون والتدفق (SFC) · Stock-flow consistent models
- دوال الاستهلاك والاستثمار والطلب على النقود، وقانون أوكن، والعجز التوأم · Consumption, investment, money demand, Okun, twin deficits

#### 2.6 اقتصاديات البيئة والمناخ والطاقة — Environment, Climate & Energy
- نماذج التقييم المتكامل للمناخ والاقتصاد: DICE وRICE وPAGE ⚙️، والتكلفة الاجتماعية للكربون · IAMs, social cost of carbon
- نماذج أنظمة الطاقة والاقتصاد (MARKAL / TIMES) ⚙️ · Energy-economy systems
- منحنى كوزنتس البيئي التطبيقي، ونماذج المناخ والإنتاجية (انظر 1.26) · EKC, climate-yield models

### ثانياً: النماذج الجزئية التطبيقية — Applied Microeconomic Models

#### 2.7 الإنتاج والتكاليف والكفاءة — Production, Cost & Efficiency
| النموذج | Model | المخرجات والرسوم |
|---|---|---|
| كوب-دوجلاس، وCES، والمتسامية اللوغاريتمية (ترانسلوج)، وليونتيف ذات النسب الثابتة، وليونتيف المعممة، والتربيعية والتربيعية المعيارية، والترانسندنتالية | Cobb-Douglas, CES, translog, Leontief, generalized Leontief, quadratic & normalized quadratic, transcendental | المرونات الإنتاجية، والعائد للسعة، ومرونات الإحلال (ألن، موريشيما)، ومنحنيات الناتج المتوسط والحدي، **ومراحل الإنتاج الثلاث**، ومنحنيات الناتج المتساوي |
| دوال استجابة المحصول للسماد: سبيلمان، وميتشرليخ، والخطي-الهضبة، والتربيعي-الهضبة | Yield response functions | **الكمية الاقتصادية المثلى** من السماد ورسم الاستجابة |
| الكميات المثلى للموارد | Optimal input use (VMP = price) | نسب قيمة الناتج الحدي إلى السعر لكل مورد |
| دالة التكاليف التكعيبية، وطويلة الأجل، ووفورات السعة والنطاق | Cubic & long-run cost, economies of scale & scope | منحنيات التكاليف، و**الحجم الأمثل**، ونقطة التعادل وإغلاق المنشأة، والحجم الأدنى الكفء |
| دوال التكاليف كوب-دوجلاس وترانسلوج مع معادلات الحصص (ليما شيفرد، SUR)، ودوال الربح | Cost functions with share equations, profit functions | مرونات الطلب على المدخلات ومرونات الإحلال والعرض |
| الحدود العشوائية (SFA) للإنتاج والتكاليف، ومحددات عدم الكفاءة (باتيس-كويلي)، وSFA للبانل، والحدود الفوقية | SFA (production & cost), Battese-Coelli, panel SFA, meta-frontier | الكفاءة الفنية والتخصيصية والاقتصادية لكل وحدة |
| DEA ومؤشر مالمكويست (انظر 5.12) | DEA & Malmquist (see 5.12) | — |
| **تقدير الإنتاجية الكلية للعوامل**: أولي-باكس، وليفينسون-بترين، وأكربرغ-كيفز-فريزر، والأرقام القياسية (تورنكفيست، وفيشر) | TFP estimation: Olley-Pakes, Levinsohn-Petrin, ACF; Törnqvist / Fisher indices | — |

#### 2.8 استجابة العرض والطلب والاستهلاك — Supply Response, Demand & Consumption
| النموذج | Model | المخرجات |
|---|---|---|
| نيرلوف (التعديل الجزئي والتوقعات المتكيفة)، واستجابة المساحة، وبيت العنكبوت، واستجابة العرض بنهج ARDL | Nerlove, acreage response, cobweb, ARDL supply response | مرونات العرض قصيرة وطويلة الأجل، ورسم تذبذب السعر والكمية |
| دالة الطلب المفردة، ونماذج المرونة الساكنة والديناميكية | Single-equation demand, static & dynamic elasticity models | المرونات السعرية والدخلية والتقاطعية |
| منحنيات إنجل (Working-Leser) | Engel curves | تصنيف السلع: ضرورية، أو كمالية، أو دنيا |
| نظام الإنفاق الخطي (LES)، وAIDS وLA-AIDS وQUAIDS، وروتردام | LES, AIDS / QUAIDS, Rotterdam | مصفوفة مرونات هيكس ومارشال |
| استهلاك دورة الحياة والدخل الدائم | Life-cycle & permanent income models | الميل الحدي للاستهلاك قصير وطويل الأجل |
| الرفاه: فائض المستهلك والتغير التعويضي والمكافئ | Consumer surplus, compensating & equivalent variation | أثر تغير الأسعار على رفاه الأسر |

#### 2.9 هياكل الأسواق والتنظيم الصناعي — Market Structure & Industrial Organization
- مقاييس التركز (هيرفندال-هيرشمان، ونسب التركز CR4 وCR8) · HHI, concentration ratios
- كورنو وبرتراند وستاكلبرج التطبيقية · Empirical Cournot, Bertrand, Stackelberg
- قوة السوق وسلوك التسعير (NEIO، بريسناهان-لاو) · New Empirical IO, Bresnahan-Lau
- بيري-ليفينسون-بايكس للطلب المتمايز (BLP) — انظر 1.6 · BLP
- التواطؤ والتسعير الديناميكي · Collusion & dynamic pricing
- المزادات الهيكلية (السعر الأول، والسعر الثاني، والقيمة المشتركة) · Structural auction models
- الدخول والخروج من السوق (بريسناهان-رايس) · Entry-exit models

#### 2.10 الأسواق والأسعار والتسويق — Markets, Prices & Marketing
- انتقال الأسعار المتماثل وغير المتماثل (NARDL، والتكامل المشترك العتبي) · Price transmission
- تكامل الأسواق وقانون السعر الواحد · Market integration
- الهوامش التسويقية ونصيب المنتج من جنيه المستهلك · Marketing margins, farmer's share
- الأرقام القياسية الموسمية، وتقلب الأسعار (ARCH/GARCH)، ومعدلات التبادل وأسعار التعادل · Seasonal indices, volatility, terms of trade, parity prices

#### 2.11 التسعير الهيدوني وتقييم السلع غير السوقية — Hedonic Pricing & Non-Market Valuation
- التسعير الهيدوني (روزن) · Hedonic pricing
- التقييم المشروط (الرغبة في الدفع والقبول) · Contingent valuation
- تجارب الاختيار المنفصل (DCE) · Discrete choice experiments
- تكلفة السفر للسلع البيئية · Travel cost models

#### 2.12 سوق العمل ورأس المال البشري والرفاهية — Labour, Human Capital & Welfare
- معادلة مينسر للأجور، وتجزئة أوكساكا-بلايندر · Mincer earnings, Oaxaca-Blinder
- البحث والمطابقة في سوق العمل · Search & matching
- عرض العمل المنفصل والمستمر · Discrete & continuous labour supply
- **المحاكاة الدقيقة للضرائب والإعانات** (على غرار EUROMOD) ⚙️ · Tax-benefit microsimulation
- الفقر (FGT) وعدم المساواة (جيني، وثايل، وأتكنسون) مع التجزئة، والمؤشرات المركبة (على غرار مؤشر التنمية البشرية) · Poverty, inequality decomposition, composite indices

#### 2.13 الاقتصاد الزراعي والموارد — Agricultural & Resource Economics
- مؤشرات عدم الاستقرار والمخاطر (كادي-ديلا فالي، وكوبوك) · Instability indices
- تبني التكنولوجيا (انظر 1.4: العقبة المزدوجة) · Technology adoption
- النهج الريكاردي واستجابة الإنتاجية للمناخ (انظر 1.26) · Ricardian & climate response
- الميزانيات المزرعية والهوامش الإجمالية · Farm budgets & gross margins

#### 2.14 الاقتصاد المالي — Financial Economics
- CAPM ونماذج فاما-فرنش، ودراسة الحدث · CAPM, Fama-French, event study
- القيمة المعرضة للخطر، والمحفظة المثلى، والحد الكفء لماركوفيتز · VaR, Markowitz efficient frontier

### ثالثاً: النماذج الاقتصادية التطبيقية بالذكاء الاصطناعي — AI-Driven Applied Economics 🧠

#### 2.15 الكلي والهيكلي — Macro & structural
- المحاكاة الاقتصادية القائمة على الوكلاء الأذكياء (AI-ABM) · AI-driven agent-based models
- التعلم المعزز للتوازن الاقتصادي والسياسة الضريبية («الاقتصادي الآلي» AI Economist، MARL) · AI Economist
- التوازن العام العميق (Deep DSGE، وDeep HANK) · Deep equilibrium models
- التوأم الرقمي للاقتصاد وسلاسل الإمداد · Economic & supply-chain digital twins
- المحاكاة السلوكية التوليدية بوكلاء النماذج اللغوية الكبيرة (LLM Agents) · Generative agent simulations
- تقدير الفقر والناتج المحلي من صور الأقمار الصناعية · Satellite-AI economic mapping
- **الإنذار المبكر بالأزمات المالية والنظامية بالذكاء الاصطناعي** · AI early-warning systems

#### 2.16 الجزئي — Micro
- التسعير الخوارزمي والتواطؤ الذكي · Algorithmic pricing & AI collusion
- تقدير مرونات الطلب بالتعلم العميق (DeepIV، وNeural-BLP) · Deep demand & elasticity estimation
- تصميم الأسواق والمزادات والآليات بالذكاء الاصطناعي · AI market, auction & mechanism design
- المحاكاة السلوكية للمستهلكين والمنشآت بالوكلاء الأذكياء · AI agent-based micro simulations

---

## 3. محور التنبؤ الاقتصادي والمالي — Economic & Financial Forecasting

### 3.1 النماذج الإحصائية والكلاسيكية — Statistical & Classical
- الساذج، والساذج الموسمي، والمتوسطات المتحركة · Naive, seasonal naive, moving averages
- التمهيد الأسي (البسيط، وهولت، وهولت-وينترز، وإطار ETS) · Exponential smoothing
- الفضاء الحالتي ومرشح كالمان · State-space models & Kalman filter
- ARIMA وSARIMA وSARIMAX، مع اختيار الرتبة تلقائياً · ARIMA / SARIMA / SARIMAX
- نموذج ثيتا · Theta
- السلاسل الزمنية الهيكلية البايزية (BSTS) · Bayesian structural time series
- نماذج الاتجاه العام ومنحنيات النمو (الخطي، والتربيعي، والأسي، واللوجستي، وجومبرتز) · Trend & growth curves

### 3.2 التنبؤ الاقتصادي الكلي والآني — Macro & Nowcasting
- **التنبؤ الآني** (معادلات الجسر، وMIDAS) · Nowcasting: bridge equations, MIDAS
- العوامل الديناميكية (DFM)، وBVAR، وVAR وVECM وARDL · DFM, BVAR, VAR / VECM / ARDL
- الإسقاطات المحلية للتنبؤ متعدد الآفاق · Local projections
- **تجميع التنبؤات** (المتوسط، والأوزان المثلى، والتجميع البايزي) · Forecast combination / ensembles

### 3.3 التنبؤ المالي والتقلب — Financial & Volatility
- التنبؤ بالتقلب (GARCH، وHAR-RV) والقيمة المعرضة للخطر · Volatility & VaR forecasting

### 3.4 التنبؤ بالتعلم الآلي — Machine-Learning Forecasting
- تعزيز التدرج (XGBoost، وLightGBM، وCatBoost) للسلاسل الزمنية · Gradient boosting
- الغابات العشوائية، وآلات المتجهات الداعمة (SVR) · Random forest, SVR
- Prophet وNeuralProphet · Prophet / NeuralProphet

### 3.5 التنبؤ بالذكاء الاصطناعي والتعلم العميق — AI & Deep Learning 🧠
- الشبكات العصبية المتكررة (RNN، وLSTM، وBi-LSTM، وGRU) · Recurrent networks
- المحولات للسلاسل الزمنية (TFT، وInformer، وAutoformer، وPatchTST) · Transformers
- N-BEATS وN-HiTS · Neural basis expansion
- التنبؤ العميق الاحتمالي (DeepAR) · DeepAR
- الخزان الحوسبي (Echo State Networks) · Reservoir computing
- **النماذج التأسيسية للسلاسل الزمنية** (TimesFM، وChronos، وLag-Llama، وTimeGPT) · Time-series foundation models
- النماذج الهجينة (ARIMA-ANN، وGARCH-LSTM) — انظر 1.23 · Hybrid models

### 3.6 تقييم التنبؤ وعرضه — Evaluation & Presentation (لكل النماذج)
- مقاييس الدقة: MAE، وRMSE، وMAPE، وsMAPE، وMASE، ومعامل ثايل U · Accuracy metrics
- اختبار ديبولد-ماريانو، والاختبار التاريخي بالأصل المتدحرج · Diebold-Mariano, rolling-origin backtesting
- فترات التنبؤ، ورسم المروحة، و**التنبؤ بالسيناريوهات لصانع القرار** · Prediction intervals, fan charts, scenario forecasting
- **مسابقة النماذج**: تشغيل عدة نماذج على السلسلة نفسها وترتيبها حسب الدقة · Automatic model horse-race

---

## 4. محور التحليل متعدد المتغيرات وعلم البيانات — Multivariate & Data Science
- المكونات الرئيسية والتحليل العاملي (الاستكشافي والتوكيدي) · PCA, EFA, CFA
- التحليل العنقودي (k-means، الهرمي) والتمييزي · cluster & discriminant analysis
- تحليل التطابق، وMANOVA، والارتباط القانوني · correspondence analysis, MANOVA, canonical correlation
- **نمذجة المعادلات البنائية (SEM) وتحليل المسار** · structural equation modelling, path analysis
- الثبات والصدق للاستبيانات (ألفا كرونباخ، أوميجا، الصدق التقاربي والتمييزي) · reliability & validity
- التعلم الآلي: أشجار القرار، والغابات العشوائية، وXGBoost، وSVM، والشبكات العصبية، مع **تفسير SHAP** · ML with explainability

## 5. محور بحوث العمليات — Operations Research
> لكل نموذج: صياغة النموذج من بيانات المستخدم، والحل الأمثل، و**خطوات الحل التفصيلية للطلاب** (حيث ينطبق)، وتحليل الحساسية، والرسوم (المنطقة الممكنة، والشبكات، ومخطط جانت، ومنحنيات المحاكاة)، والتفسير الاقتصادي، وتوصية لصانع القرار.

### 5.1 البرمجة الخطية — Linear Programming
| النموذج / الطريقة | Model / method |
|---|---|
| الصياغة الرياضية من مسألة لفظية (مع مساعد) | Problem formulation assistant |
| الحل البياني لمتغيرين مع رسم المنطقة الممكنة وخط دالة الهدف | Graphical method |
| طريقة السمبلكس بجداولها خطوة بخطوة، والمعاقبة الكبرى (Big-M)، وطريقة المرحلتين | Simplex tableaux, Big-M, two-phase |
| السمبلكس الثنائي والمعدّل، والمسألة الثنائية وتفسيرها الاقتصادي | Dual simplex, revised simplex, duality |
| تحليل الحساسية: مدى معاملات دالة الهدف والطرف الأيمن، **أسعار الظل**، والتكاليف المخفضة، والبرمجة البارامترية | Sensitivity, shadow prices, reduced costs, parametric LP |
| الحالات الخاصة: الحلول المتعددة، وغير المحدود، وغير الممكن، والانحلال | Special cases |
| البرمجة الصحيحة والمختلطة: الفرع والحد، والمستويات القاطعة، ومسائل الحقيبة 0-1، والتكلفة الثابتة، وتغطية المجموعات، وتحديد مواقع المرافق | Integer / mixed-integer, B&B, cutting planes, knapsack, fixed charge, set covering, facility location |
| تطبيقات: التخطيط المزرعي الأمثل، وأقل تكلفة لعلائق الحيوان، ومزيج الإنتاج، وتخطيط القوى العاملة، والخلط | Farm planning, least-cost ration, product mix, staffing, blending |

### 5.2 البرمجة غير الخطية والطرق العددية — Nonlinear & Numerical Programming
| النموذج / الطريقة | Model / method |
|---|---|
| الأمثلية غير المقيدة: الشروط الضرورية والكافية، ونيوتن-رافسون، والتدرج (أشد انحدار)، وطرق شبه نيوتن (BFGS) | Unconstrained optimisation, Newton, gradient, BFGS |
| البحث أحادي البعد: النسبة الذهبية، وفيبوناتشي، والتنصيف | Line search: golden section, Fibonacci, bisection |
| الأمثلية المقيدة: مضاعفات لاجرانج، وشروط كاروش-كون-تاكر (KKT) | Lagrange multipliers, KKT conditions |
| البرمجة التربيعية (ومنها محفظة ماركوفيتز وMOTAD)، والبرمجة القابلة للفصل والكسرية والهندسية والمحدبة | Quadratic, separable, fractional, geometric, convex programming |
| الخوارزميات ما وراء الاستدلالية: انظر المحور 5.15 | Metaheuristics: see 5.15 |

### 5.3 برمجة الأهداف وتعدد الأهداف — Goal & Multi-objective Programming
| النموذج / الطريقة | Model / method |
|---|---|
| برمجة الأهداف الموزونة والتسلسلية (ذات الأولويات) والحد الأدنى-الأعظم | Weighted, preemptive (lexicographic), min-max (Chebyshev) goal programming |
| برمجة الأهداف الضبابية | Fuzzy goal programming |
| البرمجة متعددة الأهداف: المجموع الموزون، وقيد إبسلون، و**رسم حدود باريتو** | Weighted sum, ε-constraint, Pareto frontier |
| التطبيقات: الموازنة بين الربح والمخاطرة والمياه والعمالة في الخطط الزراعية، والتخطيط الإقليمي | Profit–risk–water–labour trade-offs, regional planning |

### 5.4 نماذج اتخاذ القرار متعدد المعايير — MCDM
| النموذج / الطريقة | Model / method |
|---|---|
| العملية الهرمية التحليلية AHP، مع **نسبة الاتساق** ودمج آراء الخبراء (المتوسط الهندسي) | AHP with consistency ratio, group aggregation |
| AHP الضبابي، والشبكية ANP | Fuzzy AHP, ANP |
| TOPSIS، وTOPSIS الضبابي | TOPSIS, fuzzy TOPSIS |
| VIKOR، وPROMETHEE I/II، وELECTRE، وSAW، وWASPAS، وCOPRAS، وMOORA، وEDAS | VIKOR, PROMETHEE, ELECTRE, SAW, WASPAS, COPRAS, MOORA, EDAS |
| أساليب الأوزان: الإنتروبي، وCRITIC، وأفضل-أسوأ (BWM)، وSWARA | Entropy, CRITIC, Best-Worst Method, SWARA weights |
| DEMATEL للعلاقات السببية بين المعايير | DEMATEL |
| **تحليل حساسية الترتيب** لتغير الأوزان، والمقارنة بين نتائج الطرق المختلفة | Rank sensitivity & method comparison |

### 5.5 نماذج النقل والتخصيص — Transportation & Assignment
| النموذج / الطريقة | Model / method |
|---|---|
| الحل الأساسي الأولي: الركن الشمالي الغربي، وأقل تكلفة، و**فوجل التقريبية** | NW corner, least cost, Vogel's approximation |
| اختبار الأمثلية: MODI (u-v) والحجر المتنقل | MODI, stepping stone |
| الحالات الخاصة: غير المتوازنة، والانحلال، والمسارات الممنوعة، وتعظيم الربح | Unbalanced, degeneracy, prohibited routes, maximisation |
| مسائل الشحن العابر (Transshipment) | Transshipment |
| التخصيص: **الطريقة المجرية**، وغير المتوازن، والتعظيم، والقيود | Hungarian method and variants |
| البائع المتجول وتوجيه المركبات: انظر المحور 5.16 | TSP & VRP: see 5.16 |

### 5.6 نظرية الألعاب ونماذج التفاعل الاستراتيجي — Game Theory & Strategic Interaction
| النموذج / الطريقة | Model / method |
|---|---|
| المباريات الصفرية لشخصين: نقطة السرج، والسيطرة، والاستراتيجيات المختلطة (جبرياً، وبيانياً، وبالبرمجة الخطية) | Two-person zero-sum, saddle point, dominance, mixed strategies |
| المباريات غير الصفرية: **توازن ناش**، ومعضلة السجين، والمباريات المتسلسلة (شجرة اللعبة والاستقراء العكسي) | Nash equilibrium, prisoner's dilemma, sequential games, backward induction |
| نماذج احتكار القلة: كورنو، وبرتراند، وستاكلبرج، والتواطؤ | Cournot, Bertrand, Stackelberg, collusion |
| المباريات التعاونية: **قيمة شابلي**، والنواة، والمساومة لناش (تقاسم المياه والموارد) | Shapley value, core, Nash bargaining |
| المباريات المتكررة والتطورية والمعلومات غير الكاملة (Bayesian) | Repeated, evolutionary, Bayesian games |
| تطبيقات: التفاوض التجاري، وإدارة الموارد المشتركة، وسياسات التسعير | Trade negotiation, common-pool resources, pricing |

### 5.7 نماذج الشبكات وإدارة المشروعات — Networks & Project Management
| النموذج / الطريقة | Model / method |
|---|---|
| أقصر مسار (ديكسترا، فلويد)، وأصغر شجرة ممتدة (بريم، كروسكال)، والتدفق الأقصى (فورد-فولكرسون)، والتدفق الأقل تكلفة | Shortest path, minimum spanning tree, max flow, min-cost flow |
| **CPM**: المسار الحرج، والأوقات المبكرة والمتأخرة، والفائض الكلي والحر، و**مخطط جانت** | Critical path, floats, Gantt chart |
| **PERT**: التقديرات الثلاثة، و**احتمال إنجاز المشروع في موعد محدد** | Three-point estimates, completion probability |
| ضغط المشروع (الوقت-التكلفة)، وتسوية الموارد وتحميلها | Crashing, resource levelling & loading |
| محاكاة مونت كارلو لمدة المشروع | Monte Carlo schedule risk |

### 5.8 نماذج المخزون — Inventory
| النموذج / الطريقة | Model / method |
|---|---|
| الحتمية: EOQ، والإنتاج الاقتصادي EPQ، والعجز المسموح، وخصم الكمية، والأصناف المتعددة بقيود | EOQ, EPQ, planned shortages, quantity discounts, multi-item constrained |
| الاحتمالية: نقطة إعادة الطلب ومخزون الأمان، ونظاما (s, S) و(R, Q)، والمراجعة الدورية | Reorder point & safety stock, (s,S), (R,Q), periodic review |
| نموذج بائع الصحف (الفترة الواحدة) | Newsvendor |
| تحليل ABC وVED | ABC / VED analysis |
| تطبيقات: مخزون الحبوب والأعلاف والأسمدة، والمخزون الاستراتيجي من السلع | Grain, feed, fertiliser & strategic reserves |

### 5.9 نماذج صفوف الانتظار — Queueing
| النموذج / الطريقة | Model / method |
|---|---|
| M/M/1، وM/M/c، وM/M/1/K، وM/M/c/K، والمجتمع المحدود | M/M/1, M/M/c, finite capacity & finite source |
| M/G/1 (بولاتشيك-خينتشين)، وM/D/1، وأولويات الخدمة | M/G/1, M/D/1, priority queues |
| مقاييس الأداء: Lq وL وWq وW وP0 ونسبة الاستخدام | Performance measures |
| **اقتصاديات صفوف الانتظار**: العدد الأمثل لمنافذ الخدمة الذي يقلل التكلفة الكلية | Cost optimisation of servers |
| تطبيقات: صوامع الاستلام، والموانئ، والمستشفيات، والبنوك | Silos, ports, hospitals, banks |

### 5.10 المحاكاة وتحليل القرار — Simulation & Decision Analysis
| النموذج / الطريقة | Model / method |
|---|---|
| توليد الأعداد العشوائية واختبارها، وتوليد المتغيرات من التوزيعات | Random number & variate generation |
| **محاكاة مونت كارلو** مع تحليل المخاطر للمشروعات والأرباح الزراعية | Monte Carlo risk analysis |
| محاكاة الأحداث المتقطعة (صفوف الانتظار والمخزون) | Discrete-event simulation |
| القرار في حالة عدم التأكد: لابلاس، وماكسيمين، وماكسيماكس، وهورويتز، وسافاج (الندم الأدنى) | Laplace, maximin, maximax, Hurwicz, Savage |
| القرار في حالة المخاطرة: القيمة النقدية المتوقعة، والخسارة المتوقعة، و**شجرة القرار**، وقيمة المعلومات الكاملة والعينية (بيز) | EMV, EOL, decision trees, EVPI, EVSI (Bayesian) |
| نظرية المنفعة ودرجة تقبّل المخاطرة (المكافئ المؤكد) | Utility theory, certainty equivalent, risk attitude |
| تطبيقات زراعية للمخاطر: MOTAD وTarget MOTAD، والمتوسط-التباين، وأمان أولاً (Safety-first)، والهيمنة العشوائية | MOTAD, Target MOTAD, E-V, safety-first, stochastic dominance |

### 5.11 النماذج الديناميكية والاحتمالية — Dynamic & Stochastic Models
| النموذج / الطريقة | Model / method |
|---|---|
| **سلاسل ماركوف**: مصفوفة الانتقال، والتنبؤ بالحالات، و**الحالة المستقرة**، والسلاسل الماصّة (متوسط زمن الامتصاص) | Markov chains: transitions, steady state, absorbing chains |
| تطبيقات ماركوف: حصص السوق وولاء المستهلك، وتطور حجم المزارع، وحركة الحيازات | Market shares, brand loyalty, farm-size dynamics |
| عمليات القرار الماركوفية (MDP): تكرار القيمة وتكرار السياسة | Markov decision processes |
| **البرمجة الديناميكية**: الحقيبة، وتوزيع الموارد، وأقصر مسار متعدد المراحل، وجدولة المخزون والإنتاج، واستبدال المعدات | Dynamic programming: knapsack, resource allocation, staged shortest path, production–inventory, equipment replacement |
| البرمجة الديناميكية العشوائية | Stochastic dynamic programming |
| **البرمجة العشوائية**: النموذج ذو المرحلتين مع التعويض، والمتعدد المراحل، والمبنية على السيناريوهات (القيود الاحتمالية والتحسين الحصين: انظر 5.14) | Two-stage & multi-stage recourse, scenario-based (chance-constrained & robust: see 5.14) |
| تطبيقات: التخطيط الزراعي في ظل عدم تأكد الأمطار والأسعار والمياه | Planning under rainfall, price & water uncertainty |

### 5.12 نماذج تقييم الكفاءة والأداء — Efficiency & Performance
| النموذج / الطريقة | Model / method |
|---|---|
| **تحليل مغلف البيانات DEA**: CCR وBCC (توجيه المدخلات والمخرجات)، وكفاءة الحجم ونوع العائد للسعة | DEA CCR / BCC, input/output orientation, scale efficiency, RTS |
| النماذج المتقدمة: SBM (المعتمد على الفوائض)، والكفاءة الفائقة، والكفاءة التقاطعية، والمخرجات غير المرغوبة (التلوث) | SBM, super-efficiency, cross-efficiency, undesirable outputs |
| كفاءة التكاليف والإيرادات والتخصيصية | Cost, revenue & allocative efficiency |
| DEA عبر الزمن: **مؤشر مالمكويست** وتجزئته، وتحليل النافذة | Malmquist index, window analysis |
| DEA الشبكي (متعدد المراحل) | Network DEA |
| **DEA بالبوتستراب** (Simar-Wilson) لفترات الثقة، ثم تحليل المرحلة الثانية (توبيت، والانحدار المبتور) لمحددات الكفاءة | Bootstrap DEA, second-stage Tobit / truncated regression |
| المقارنة المرجعية: الوحدات المرجعية (Peers) والقيم المستهدفة للتحسين | Benchmarking: peers & improvement targets |
| تحليل الحدود العشوائية SFA (انظر 2.7) والمقارنة بين DEA وSFA | SFA cross-reference & DEA–SFA comparison |
| مؤشرات الأداء المركبة وبطاقة الأداء المتوازن | Composite KPIs, balanced scorecard |

### 5.13 الأنظمة المعقدة — Complex Systems
| النموذج / الطريقة | Model / method |
|---|---|
| **ديناميكيات النظم** (System Dynamics): مخططات الحلقات السببية، ونماذج المخزون والتدفق، والمحاكاة عبر الزمن | Causal loop diagrams, stock-and-flow models, simulation |
| اختبار السياسات بديناميكيات النظم: الأمن الغذائي، والمياه-الغذاء-الطاقة، والنمو السكاني، وسلاسل الإمداد (أثر السوط) | Food security, water–food–energy nexus, population, bullwhip effect |
| تحليل الحساسية واختبار صحة النموذج (الظروف القصوى، وإعادة إنتاج السلوك التاريخي) | Sensitivity & validation tests |
| **المحاكاة القائمة على الوكلاء** (Agent-Based Modelling): وكلاء بقواعد سلوك، وبيئة، وظهور السلوك الجمعي | Agents, rules, environment, emergence |
| تطبيقات الوكلاء: انتشار تبني التكنولوجيا بين المزارعين، وأسواق السلع، واستخدام الأراضي، وانتشار الأوبئة، ونموذج شيلينج | Technology diffusion, commodity markets, land use, epidemics, Schelling |
| تكرارات مونت كارلو لنماذج الوكلاء، وتحليل الحساسية للمعلمات | Monte Carlo replications, parameter sweeps |
| تحليل الشبكات الاجتماعية والاقتصادية (المركزية، والمجتمعات)، والأوتوماتا الخلوية | Network analysis (centrality, communities), cellular automata |

### 5.14 التحسين المرن وغير الدقيق — Fuzzy & Robust Optimization
| النموذج / الطريقة | Model / method |
|---|---|
| **التحسين الحصين** (Robust Optimization): مجموعات عدم التأكد الصندوقية والإهليلجية والموازنة (Bertsimas-Sim) | Robust optimisation: box, ellipsoidal, budgeted uncertainty sets |
| **البرمجة الرياضية الضبابية**: البرمجة الخطية الضبابية (Zimmermann)، والمعاملات الضبابية، والقيود المرنة | Fuzzy LP (Zimmermann), fuzzy coefficients, soft constraints |
| **التحسين الحصين الموزع** (DRO): مجموعات الغموض بالعزوم وبمسافة واسرشتاين | Distributionally robust: moment-based & Wasserstein ambiguity sets |
| **البرمجة بالقيود الاحتمالية** (Chance-Constrained): الفردية والمشتركة، والتقريب بالعينة (SAA) | Individual & joint chance constraints, sample average approximation |
| مقارنة الحلول: الحتمي، والعشوائي، والحصين (ثمن الحصانة وقيمة الحل العشوائي VSS) | Deterministic vs stochastic vs robust: price of robustness, VSS |

### 5.15 الخوارزميات ما وراء الاستدلالية والتحسين التطوري — Metaheuristics & Evolutionary Optimization
| النموذج / الطريقة | Model / method |
|---|---|
| الخوارزميات الجينية (GA) وNSGA-II لتعدد الأهداف | Genetic algorithms, NSGA-II |
| تحسين سرب الجسيمات (PSO) | Particle swarm optimisation |
| تحسين مستعمرة النمل (ACO) | Ant colony optimisation |
| التلدين المحاكى (SA) | Simulated annealing |
| البحث المحظور (Tabu Search) | Tabu search |
| التطور التفاضلي (DE) | Differential evolution |
| أخرى: البحث المحلي التكراري، والجوار المتغير (VNS)، ونحل العسل الاصطناعي (ABC)، وخوارزمية الذئب الرمادي (GWO) | ILS, VNS, artificial bee colony, grey wolf optimiser |
| لوحة المقارنة: منحنيات التقارب، وأفضل حل وأسوئه ومتوسطه عبر التكرارات، وضبط المعلمات | Convergence curves, multi-run statistics, parameter tuning |

### 5.16 تحسين الشبكات وسلاسل الإمداد المتقدمة — Advanced Network & Supply Chain
| النموذج / الطريقة | Model / method |
|---|---|
| **توجيه المركبات** (VRP) بقيود السعة (CVRP) والنوافذ الزمنية (VRPTW)، والبائع المتجول (TSP) | VRP, CVRP, VRPTW, TSP |
| **تحديد مواقع التسهيلات**: P-Median، وP-Center، وتغطية المجموعات، والتغطية القصوى، والمواقع غير المقيدة السعة | Facility location: p-median, p-center, set covering, maximal covering, UFLP |
| التدفق الأقصى وأقصر مسار والتدفق الأقل تكلفة على الشبكات الكبيرة (انظر 5.7) | Max flow, shortest path, min-cost flow (see 5.7) |
| **التحسين ثنائي المستوى** (القائد-التابع): سياسات الدعم والتسعير والضرائب | Bi-level (Stackelberg) optimisation: subsidy, pricing, tax policy |
| **التخطيط والجدولة المتكاملة وأحجام الدفعات**: Wagner-Whitin، والمسائل ذات السعة (CLSP)، وجدولة الورش (Job-shop / Flow-shop) | Lot-sizing (Wagner-Whitin, CLSP), job-shop / flow-shop scheduling |
| تصميم سلاسل الإمداد، وتوجيه المخزون، وسلاسل الإمداد الغذائية وسلاسل التبريد | Supply-chain network design, inventory routing, food & cold chains |

### 5.17 بحوث العمليات المعتمدة على الذكاء الاصطناعي — AI-Driven Operations Research 🧠
| النموذج / الطريقة | Model / method |
|---|---|
| التحسين التوافقي بالتعلم المعزز (RL4CO) | Reinforcement learning for combinatorial optimisation |
| **التنبؤ ثم التحسين** و«التنبؤ ثم التحسين الذكي» (SPO / SPO+) | Predict-then-optimise, Smart Predict-then-Optimise |
| التحسين المعزز بالتعلم الآلي: تعلّم التفرع، والبدايات الدافئة، وتعلّم القيود | ML-augmented optimisation: learning to branch, warm starts, constraint learning |
| التعلم المعزز متعدد الوكلاء لبحوث العمليات (MARL) | Multi-agent reinforcement learning for OR |
| الشبكات العصبية البيانية للتحسين التوافقي (GNN) | Graph neural networks for combinatorial optimisation |
| **التحسين البايزي** (العمليات الغاوسية ودوال الاكتساب) | Bayesian optimisation |
| التحسين القائم على المحاكاة بالنماذج البديلة (Surrogate) | Simulation-based optimisation with surrogate models |

## 6. محور تقييم المشروعات والتحليل المالي — Project Appraisal
- صافي القيمة الحالية، ومعدل العائد الداخلي، ونسبة المنافع إلى التكاليف، وفترة الاسترداد · NPV, IRR, B/C, payback
- تحليل الحساسية والتعادل، والتحليل الاقتصادي بأسعار الظل · sensitivity, switching values, economic analysis
- جداول استهلاك القروض · loan amortization

## 7. محور الأرقام القياسية — Index Numbers
- لاسبير، وباشي، وفيشر، وتورنكفيست، والأرقام المتسلسلة · Laspeyres, Paasche, Fisher, Törnqvist, chain indices
- تحويل السلاسل من الأسعار الجارية إلى الثابتة، وتغيير سنة الأساس · deflating, rebasing

---

## 8. المحاور العرضية — Cross-cutting
| المحور | المحتوى |
|---|---|
| 🏛️ **لوحة صانع القرار** | ملخص بلغة غير فنية، ومحاكاة السيناريوهات، ومؤشرات الإنذار المبكر |
| 📄 **استوديو التقارير** | تجميع النتائج في تقرير، وتصدير Word وPDF وLaTeX، والجداول بنمط APA والمجلات |
| 🤖 **مساعد الذكاء الاصطناعي** | يقترح النموذج، ويكتب الفقرات المنهجية وفقرات النتائج **من الأرقام المحسوبة فعلاً**، ويجيب عن أسئلة الباحث بالعربية |
| 🎓 **الأكاديمية** | مسار تعليمي لكل محور: شرح النظرية، ثم المعادلات، ثم مثال محلول، ثم تمرين على بيانات حقيقية، ثم اختبار وشهادة |
| 🔁 **قابلية إعادة الإنتاج** | تصدير كود Python وR وStata لكل تحليل |
| ✔️ **ضمان الجودة** | مقارنة كل نموذج بمخرجات Stata وR وEViews المنشورة قبل إطلاقه |

---

## 9. ترتيب التنفيذ المقترح — Delivery order
| الإصدار | المحتوى |
|---|---|
| v1.4 ✅ | 1.1 الانحدار الكلاسيكي والمقطعي + 1.3 التشخيص الكامل + جداول المجلات |
| v1.5 🔜 | 1.2 المعادلات الآنية وIV + 1.4 الاستجابة النوعية + 1.5 العدّ والبقاء |
| v1.6 | 1.7 السلاسل الزمنية الأحادية + جذور الوحدة |
| v1.7 | 1.8 المتعددة والتكامل المشترك (VAR، VECM، ARDL، NARDL) + 1.9 التقلب |
| v1.8 | 1.15 البانل كاملاً |
| v1.9 | 3.1–3.4 و3.6 التنبؤ: الكلاسيكي، والكلي والآني، والمالي، والتعلم الآلي، مع مسابقة النماذج |
| v1.10 | 2.7–2.8 الإنتاج والتكاليف والكفاءة، والعرض والطلب والاستهلاك |
| v1.11 | 1.14 الكمي + 1.20 تقييم السياسات + 1.16 المكاني + 1.26 المناخ والطاقة |
| v1.12 | 8. استوديو التقارير + مساعد الذكاء الاصطناعي + لوحة صانع القرار |
| v2.0 – v2.0.8 | 5.1–5.13 بحوث العمليات (ثلاثة إصدارات) + 6. تقييم المشروعات |
| v2.0.9 | 5.14 التحسين الحصين والضبابي + 5.15 الخوارزميات ما وراء الاستدلالية + 5.16 الشبكات وسلاسل الإمداد المتقدمة |
| v2.1 | 2.2–2.5 المدخلات والمخرجات، والنمو، والتجارة، والكلي والسياسات + 2.9–2.14 الأسواق والتنظيم الصناعي والتقييم والعمل والزراعة والمالية + 7. الأرقام القياسية |
| v2.2 | 1.10–1.13 عالية التردد، والعوامل والأبعاد العالية، والمعاملات المتغيرة، واللامعلمي + 4. متعدد المتغيرات وSEM |
| v2.3 | 1.6 الخيارات الهيكلية + 1.17 الشبكات والعدوى + 1.18 البايزي + 1.21 التعلم الآلي السببي |
| v2.3.5 ⚙️ | 2.1 التوازن العام (CGE وDCGE وOLG والتوازن الجزئي) + 2.4 GTAP + 2.5 النماذج الكلية واسعة النطاق + 2.6 IAMs وMARKAL/TIMES + 2.12 المحاكاة الدقيقة |
| v2.4 🧠 | 2.15–2.16 الاقتصاد التطبيقي بالذكاء الاصطناعي + 3.5 التنبؤ العميق + 5.17 بحوث العمليات بالذكاء الاصطناعي + 1.19 DSGE وHANK + 1.22 العميق + 1.23 التنبؤ بالذكاء الاصطناعي + 1.24 النصوص والصور + 1.25 تحسين السياسات |
| v2.5+ | الأكاديمية والمجتمع، والحسابات والاشتراكات |

> ⚙️ **النماذج الهيكلية الكبيرة** (CGE وGTAP وMARKAL/TIMES وIAMs وOLG) تحتاج محركات حل للمعادلات غير الخطية الكبيرة (مثل IPOPT عبر Pyomo)،
> وبعضها يحتاج قواعد بيانات بترخيص خاص (مثل قاعدة GTAP). ستُقدَّم بنماذج معايرة جاهزة قابلة للتعديل، مع إمكان رفع مصفوفة SAM أو بيانات الباحث.
>
> 🧠 **ملاحظة البنية التحتية:** نماذج التعلم العميق والتعلم المعزز وDSGE وHANK وتحليل صور الأقمار الصناعية تحتاج مكتبات ثقيلة (مثل PyTorch) وقدرة حوسبة،
> وهي تتجاوز إمكانات الاستضافة المجانية لـ Streamlit. قبل الإصدار v2.4 سننقل الحسابات الثقيلة إلى خادم مستقل (API)، تبقى
> الواجهة على Streamlit وتُرسل إليه المهام الثقيلة. بقية النماذج كلها تعمل على الاستضافة الحالية.
