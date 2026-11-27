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

## 1. محور الانحدار — Regression Lab

### 1.1 الانحدار الخطي وصوره
| النموذج | Model | الحالة |
|---|---|---|
| الانحدار البسيط والمتعدد (OLS) | Simple & multiple OLS | 🔜 |
| الانحدار المتعدد الحدود (تربيعي، تكعيبي) | Polynomial regression | 🔜 |
| الانحدار التدريجي (أمامي، خلفي، أفضل مجموعة جزئية) | Stepwise / forward / backward / best subsets | 🔜 |
| الانحدار الهرمي (إدخال المتغيرات على كتل) | Hierarchical (blockwise) regression | 🔜 |
| الصور الدالية: خطي، لوغاريتمي مزدوج، نصف لوغاريتمي، مقلوب، أسي، تحويل Box-Cox | Functional forms & Box-Cox | 🔜 |
| المتغيرات الصورية وتحليل التغاير (ANCOVA) | Dummy variables / ANCOVA | 🔜 |
| التفاعل والتأثير المعدِّل (Moderation) مع رسم الميول البسيطة | Interaction / moderation, simple slopes | 🔜 |
| التأثير الوسيط (Mediation): بارون وكيني، سوبل، بوتستراب الأثر غير المباشر | Mediation (Baron-Kenny, Sobel, bootstrap) | ⬜ |
| المربعات الصغرى الموزونة والمعممة (WLS, GLS, FGLS) | WLS / GLS / FGLS | 🔜 |
| الانحدار المتين (Huber, Bisquare M-estimators) | Robust regression | ⬜ |
| الانحدار الكمي (Quantile regression) مع رسم المعاملات عبر الكميات | Quantile regression | ⬜ |
| انحدار ريدج ولاسو والشبكة المرنة | Ridge / Lasso / Elastic net | ⬜ |
| انحدار المكونات الرئيسية والمربعات الصغرى الجزئية | PCR / PLS | ⬜ |
| المربعات الصغرى غير الخطية (NLS) | Nonlinear least squares | ⬜ |
| الانحدار المجزأ والعتبي والشرائح (Piecewise / threshold / splines / GAM) | Segmented, threshold, spline, GAM | ⬜ |
| المربعات الصغرى المقيدة (اختبار ثبات العائد للسعة) | Restricted LS (e.g. constant returns) | 🔜 |

### 1.2 أنظمة المعادلات والمتغيرات الأداتية
| النموذج | Model | الحالة |
|---|---|---|
| المتغيرات الأداتية 2SLS، LIML، GMM مع اختبارات ضعف الأدوات وفرط التحديد (Sargan / Hansen) | IV: 2SLS, LIML, GMM + weak-IV & over-id tests | ⬜ |
| اختبار ووـهاوسمان للداخلية | Durbin-Wu-Hausman endogeneity | ⬜ |
| الانحدارات غير المرتبطة ظاهرياً (SUR) | Seemingly unrelated regressions | ⬜ |
| المعادلات الآنية (2SLS، 3SLS) ومسألة التعرّف | Simultaneous equations, identification | ⬜ |

### 1.3 التشخيص الكامل (يُطبق على كل نماذج الانحدار)
- **عدم تجانس التباين:** بروش-باجان، وايت، جليجسر، بارك، جولدفيلد-كواندت · Breusch-Pagan, White, Glejser, Park, Goldfeld-Quandt
- **الارتباط الذاتي:** دربن-واتسون، بروش-جودفري، ليونج-بوكس، دربن h · DW, Breusch-Godfrey, Ljung-Box, Durbin's h
- **الازدواج الخطي:** VIF، دليل الحالة، تحليل نسب التباين · VIF, condition index, variance proportions
- **الشكل الدالي والاستقرار:** RESET، شاو، CUSUM وCUSUMSQ، المعاملات المتتابعة · RESET, Chow, CUSUM/CUSUMSQ, recursive coefficients
- **الطبيعية:** جارك-بيرا ورسم Q-Q للبواقي · Jarque-Bera, residual Q-Q
- **المشاهدات المؤثرة:** مسافة كوك، الرفع، DFBETAS، DFFITS · Cook's D, leverage, DFBETAS, DFFITS
- **الأخطاء المعيارية المتينة:** HC0–HC3، نيوي-ويست (HAC)، العنقودية، البوتستراب · HC0–HC3, Newey-West, clustered, bootstrap
- **اختبارات القيود والمفاضلة:** والد، نسبة الإمكان، مضاعف لاجرانج، J-test، AIC/BIC · Wald, LR, LM, J-test, information criteria

### 1.4 المتغير التابع المحدود — Limited dependent variables
| النموذج | Model | الحالة |
|---|---|---|
| لوجيت وبروبيت (ثنائي) مع الآثار الحدية ونسب الأرجحية ومنحنى ROC | Logit / Probit + marginal effects, odds ratios, ROC/AUC | ⬜ |
| Complementary log-log | Cloglog | ⬜ |
| لوجيت وبروبيت متعدد الحدود والمرتب | Multinomial & ordered logit / probit | ⬜ |
| لوجيت الشرطي والمتداخل (اختيار المستهلك) | Conditional & nested logit | ⬜ |
| توبيت والانحدار المبتور | Tobit (censored), truncated regression | ⬜ |
| نموذج هيكمان لتحيز الاختيار (خطوتان، الإمكان الأعظم) | Heckman selection | ⬜ |
| نموذج العقبة المزدوجة (Cragg) — تبني التكنولوجيا الزراعية | Double hurdle | ⬜ |
| نماذج العد: بواسون، ذو الحدين السالب، المضخم للأصفار، العقبة | Count: Poisson, NegBin, zero-inflated, hurdle | ⬜ |
| انحدار الاستجابة الكسرية (Papke-Wooldridge) | Fractional response | ⬜ |
| تحليل البقاء: كابلان-ماير، كوكس، ويبل | Survival: Kaplan-Meier, Cox, Weibull | ⬜ |

---

## 2. محور النماذج الاقتصادية التطبيقية — Applied Economic Models
> لكل نموذج: التقدير + المرونات + الرسوم الاقتصادية الخاصة به + التفسير الاقتصادي + المحاكاة لصانع القرار.

### 2.1 دوال الإنتاج — Production
| النموذج | Model | المخرجات والرسوم |
|---|---|---|
| كوب-دوجلاس، CES، ترانسلوج، التربيعية، ليونتيف المعممة، الترانسندنتالية | Cobb-Douglas, CES, translog, quadratic, generalized Leontief, transcendental | المرونات الإنتاجية، والعائد للسعة، ومرونات الإحلال (ألن، موريشيما)، ومنحنيات الناتج المتوسط والحدي، **ومراحل الإنتاج الثلاث**، وخريطة منحنيات الناتج المتساوي |
| دوال استجابة المحصول للسماد: سبيلمان، ميتشرليخ، الخطي-الهضبة، التربيعي-الهضبة | Yield response: Spillman, Mitscherlich, linear/quadratic-plateau | **الكمية الاقتصادية المثلى** من السماد عند تساوي قيمة الناتج الحدي مع السعر، ورسم الاستجابة |
| الكميات المثلى للموارد وكفاءة استخدامها | Optimal input use, VMP = price | نسب قيمة الناتج الحدي إلى السعر لكل مورد |

### 2.2 دوال التكاليف — Cost
| النموذج | Model | المخرجات والرسوم |
|---|---|---|
| دالة التكاليف التكعيبية القصيرة الأجل | Cubic short-run cost | منحنيات التكاليف الكلية والمتوسطة والحدية، و**الحجم الأمثل للإنتاج**، ونقطة التعادل وإغلاق المنشأة |
| دالة التكاليف طويلة الأجل ووفورات السعة | Long-run cost, economies of scale | الحجم الأدنى الكفء (MES) ومنحنى التكاليف طويل الأجل |
| دوال التكاليف كوب-دوجلاس وترانسلوج مع معادلات الحصص (ليما شيفرد، SUR) | Cobb-Douglas / translog cost + share equations | مرونات الطلب على المدخلات، ومرونات الإحلال، ووفورات النطاق |
| دوال الربح (التربيعية المعيارية، ترانسلوج) | Profit functions | مرونات العرض والطلب على المدخلات |

### 2.3 دوال الطلب — Demand
| النموذج | Model | المخرجات |
|---|---|---|
| دالة الطلب المفردة (خطية، لوغاريتمية مزدوجة) | Single-equation demand | المرونات السعرية والدخلية والتقاطعية |
| منحنيات إنجل (Working-Leser) | Engel curves | تصنيف السلع: ضرورية، أو كمالية، أو دنيا |
| نظام الإنفاق الخطي (LES) | Linear Expenditure System | الإنفاق الأدنى للكفاف |
| نظام الطلب شبه المثالي AIDS وLA-AIDS وQUAIDS | AIDS / LA-AIDS / QUAIDS | مصفوفة المرونات المعوَّضة (هيكس) وغير المعوَّضة (مارشال) |
| نموذج روتردام | Rotterdam model | — |
| الرفاه: فائض المستهلك والتغير التعويضي | Consumer surplus, compensating variation | أثر تغير الأسعار على رفاه الأسر (لصانع القرار) |

### 2.4 العرض واستجابة العرض — Supply
| النموذج | Model | المخرجات |
|---|---|---|
| نموذج نيرلوف (التعديل الجزئي والتوقعات المتكيفة) | Nerlove partial adjustment / adaptive expectations | مرونات العرض قصيرة وطويلة الأجل، ومعامل التعديل |
| استجابة المساحة المزروعة للأسعار | Acreage response | — |
| نموذج بيت العنكبوت | Cobweb model | رسم تذبذب السعر والكمية والاستقرار |
| استجابة العرض بنهج ARDL وتصحيح الخطأ | Supply response via ARDL / ECM | — |

### 2.5 تحليل الأسواق والأسعار — Markets & prices
- انتقال الأسعار المتماثل وغير المتماثل (NARDL، التكامل المشترك العتبي) · symmetric / asymmetric price transmission
- تكامل الأسواق وقانون السعر الواحد · market integration, law of one price
- الهوامش التسويقية ونصيب المنتج من جنيه المستهلك · marketing margins, farmer's share
- الأرقام القياسية الموسمية للأسعار · seasonal price indices
- تقلب الأسعار (ARCH/GARCH) · price volatility
- معدلات التبادل وأسعار التعادل · terms of trade, parity prices

### 2.6 التجارة الدولية — Trade
- نموذج الجاذبية (PPML) · gravity model
- الميزة النسبية الظاهرة (RCA، RSCA، لافاي) · revealed comparative advantage
- تحليل حصة السوق الثابتة · constant market share analysis
- مؤشرات التركز وعدم الاستقرار للصادرات، والتجارة البينية (جروبل-لويد) · export concentration & instability, intra-industry trade
- دالة الطلب على الواردات وانتقال أثر سعر الصرف، وشرط مارشال-ليرنر، ومنحنى J · import demand, exchange-rate pass-through, Marshall-Lerner, J-curve

### 2.7 الاقتصاد الكلي والنمو — Macro & growth
- دوال الاستهلاك والاستثمار (المعجّل) والطلب على النقود · consumption, investment, money demand
- منحنى فيليبس، وقانون أوكن، وقاعدة تايلور · Phillips curve, Okun's law, Taylor rule
- محاسبة النمو، والإنتاجية الكلية (بواقي سولو)، والتقارب (بيتا وسيجما) · growth accounting, TFP, convergence
- منحنى كوزنتس البيئي (EKC) · environmental Kuznets curve
- العجز التوأم، واستدامة الدين والمالية العامة · twin deficits, debt sustainability

### 2.8 التنمية والفقر والتوزيع — Development & welfare
- مقاييس الفقر (FGT)، والتفاوت (جيني، ثايل، أتكنسون) مع التجزئة · poverty and inequality with decomposition
- تجزئة أوكساكا-بلايندر، ومعادلة مينسر للعائد على التعليم · Oaxaca-Blinder, Mincer equation
- بناء المؤشرات المركبة (مثل مؤشر التنمية البشرية) · composite indices (HDI-type)

### 2.9 الكفاءة والإنتاجية — Efficiency & productivity
| النموذج | Model | الحالة |
|---|---|---|
| تحليل مغلف البيانات DEA: CCR، BCC، الكفاءة الفائقة، كفاءة الحجم | DEA: CCR, BCC, super-efficiency, scale efficiency | ⬜ |
| مؤشر مالمكويست وتجزئته (تغير تقني، وتغير الكفاءة) | Malmquist productivity index | ⬜ |
| تحليل الحدود العشوائية SFA: كوب-دوجلاس وترانسلوج، توزيعات نصف طبيعي ومبتور وأسي | Stochastic frontier analysis | ⬜ |
| محددات عدم الكفاءة (باتيس وكويلي 1995)، وSFA للبانل | Battese-Coelli 1995, panel SFA | ⬜ |
| الكفاءة الفنية والتخصيصية والاقتصادية، والحدود الفوقية | Technical / allocative / economic efficiency, meta-frontier | ⬜ |
| الأرقام القياسية للإنتاجية (تورنكفيست، فيشر) | Törnqvist / Fisher TFP indices | ⬜ |

### 2.10 الاقتصاد الزراعي والبيئي والمناخ — Agricultural, environmental & climate
- نماذج المناخ والإنتاجية للبانل، مع آثار الحرارة غير الخطية ودرجات الحرارة النامية · weather-yield panel models, degree days
- النموذج الريكاردي لتقييم أثر المناخ · Ricardian approach
- تحليل عدم الاستقرار والمخاطر (معامل كوبوك، مؤشر عدم الاستقرار) · instability indices (Cuddy-Della Valle, Coppock)
- تبني التكنولوجيا · technology adoption
- التقييم الاحتمالي (الرغبة في الدفع)، وتكلفة الرحلة، والتسعير الهيدوني · contingent valuation, travel cost, hedonic pricing

### 2.11 المدخلات والمخرجات — Input-output
- معكوس ليونتيف، والمضاعفات، والروابط الأمامية والخلفية · Leontief inverse, multipliers, linkages
- مضاعفات مصفوفة الحسابات الاجتماعية (SAM) · SAM multipliers

### 2.12 الاقتصاد المالي — Financial economics
- نموذج تسعير الأصول الرأسمالية (CAPM) ونماذج فاما-فرنش · CAPM, Fama-French
- دراسة الحدث · event study
- القيمة المعرضة للخطر، والمحفظة المثلى، والحد الكفء لماركوفيتز · VaR, Markowitz portfolio & efficient frontier

---

## 3. محور السلاسل الزمنية — Time Series
| المجموعة | النماذج والاختبارات |
|---|---|
| التحليل والتفكيك | التفكيك الكلاسيكي وSTL، والتعديل الموسمي، ومرشحات هودريك-بريسكوت وباكستر-كينج وكريستيانو-فيتزجيرالد |
| جذور الوحدة | ADF، PP، KPSS، DF-GLS، Ng-Perron، زيفوت-أندروز وLee-Strazicich (مع الانكسارات)، HEGY الموسمي |
| الانكسارات الهيكلية | شاو، باي-بيرون للانكسارات المتعددة، CUSUM |
| التكامل المشترك | إنجل-جرانجر، جوهانسن، **ARDL واختبار الحدود**، **NARDL غير المتماثل**، جريجوري-هانسن، FMOLS وDOLS وCCR |
| النماذج متعددة المتغيرات | VAR، وVECM، وSVAR، وBVAR، ودوال الاستجابة الفورية، وتجزئة التباين، والسببية لجرانجر وتودا-ياماموتو |
| التقلب | ARCH، GARCH، EGARCH، GJR-GARCH، GARCH-M |
| غير الخطية | العتبي (TAR)، وتحويل ماركوف (Markov switching)، ونماذج فضاء الحالة ومرشح كالمان |
| الترددات | التحليل الطيفي وتماسك المويجات (wavelet coherence) |

---

## 4. محور التنبؤ — Forecasting Lab
| المجموعة | النماذج |
|---|---|
| الأساليب البسيطة | الساذج، والساذج الموسمي، والمتوسطات المتحركة |
| التمهيد الأسي | البسيط، وهولت، وهولت-وينترز، وإطار ETS |
| بوكس-جنكنز | ARIMA، وSARIMA، واختيار الرتبة تلقائياً، وARIMAX مع متغيرات خارجية |
| نماذج الاتجاه العام | الخطي، والتربيعي، والأسي، و**منحنيات النمو اللوجستية وجومبرتز** |
| التنبؤ متعدد المتغيرات | VAR وVECM، والتنبؤ بنماذج الانحدار |
| التعلم الآلي | الغابات العشوائية، وتعزيز التدرج، والشبكات العصبية (LSTM)، ونموذج Theta |
| الدمج والتقييم | التنبؤات المجمّعة؛ MAE وRMSE وMAPE وMASE ومعامل ثايل U؛ اختبار ديبولد-ماريانو؛ الاختبار التاريخي بالأصل المتدحرج |
| العرض | فترات التنبؤ، ورسم المروحة، و**التنبؤ بالسيناريوهات لصانع القرار** |

---

## 5. محور بيانات البانل — Panel Data
| المجموعة | النماذج والاختبارات |
|---|---|
| النماذج الساكنة | التجميعي، والأثر الثابت (أحادي وثنائي الاتجاه)، والأثر العشوائي، والبيني، والفروق الأولى |
| الاختبارات | هاوسمان، ومضاعف لاجرانج لبروش-باجان، واختبار F، واعتماد المقاطع لبيساران (CD)، وولدريدج للارتباط الذاتي |
| الأخطاء المعيارية | دريسكول-كراي، وPCSE، والعنقودية، وFGLS |
| جذور الوحدة والتكامل | LLC، وIPS، وFisher-ADF، وCIPS؛ بيدروني، وكاو، وويسترلوند |
| البانل الديناميكي | أريلانو-بوند، وبلونديل-بوند (System GMM)، واختبارات هانسن وسارجان وAR(2) |
| البانل غير المتجانس | PMG، وMG، وDFE (بيساران)، وCCE |
| السببية | دوميترسكو-هورلن |

---

## 6. محور الاستدلال السببي وتقييم الأثر — Causal Inference & Impact Evaluation
- تحليل التجارب العشوائية (RCT) · randomized trials
- الفروق في الفروق (الكلاسيكية والمتدرجة: Callaway-Sant'Anna) ودراسة الحدث · DiD, staggered DiD, event study
- التحكم التركيبي · synthetic control
- انحدار الانقطاع (RDD) · regression discontinuity
- مطابقة درجة الميل (أقرب جار، النواة) والترجيح العكسي، والتقدير المزدوج المتانة · PSM, IPW, doubly robust
- المتغيرات الأداتية والأثر المحلي (LATE) · IV / LATE

---

## 7. محور التحليل متعدد المتغيرات وعلم البيانات — Multivariate & Data Science
- المكونات الرئيسية والتحليل العاملي (الاستكشافي والتوكيدي) · PCA, EFA, CFA
- التحليل العنقودي (k-means، الهرمي) والتمييزي · cluster & discriminant analysis
- تحليل التطابق، وMANOVA، والارتباط القانوني · correspondence analysis, MANOVA, canonical correlation
- **نمذجة المعادلات البنائية (SEM) وتحليل المسار** · structural equation modelling, path analysis
- الثبات والصدق للاستبيانات (ألفا كرونباخ، أوميجا، الصدق التقاربي والتمييزي) · reliability & validity
- التعلم الآلي: أشجار القرار، والغابات العشوائية، وXGBoost، وSVM، والشبكات العصبية، مع **تفسير SHAP** · ML with explainability

## 8. محور الاقتصاد القياسي المكاني — Spatial Econometrics
- موران I وLISA والخرائط · Moran's I, LISA maps
- النماذج المكانية SAR وSEM وSDM، والبانل المكاني · spatial lag / error / Durbin, spatial panels

---

## 9. محور بحوث العمليات — Operations Research
> لكل نموذج: صياغة النموذج من بيانات المستخدم، والحل الأمثل، و**خطوات الحل التفصيلية للطلاب** (حيث ينطبق)، وتحليل الحساسية، والرسوم (المنطقة الممكنة، والشبكات، ومخطط جانت، ومنحنيات المحاكاة)، والتفسير الاقتصادي، وتوصية لصانع القرار.

### 9.1 البرمجة الخطية — Linear Programming
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

### 9.2 البرمجة غير الخطية والطرق العددية — Nonlinear & Numerical Programming
| النموذج / الطريقة | Model / method |
|---|---|
| الأمثلية غير المقيدة: الشروط الضرورية والكافية، ونيوتن-رافسون، والتدرج (أشد انحدار)، وطرق شبه نيوتن (BFGS) | Unconstrained optimisation, Newton, gradient, BFGS |
| البحث أحادي البعد: النسبة الذهبية، وفيبوناتشي، والتنصيف | Line search: golden section, Fibonacci, bisection |
| الأمثلية المقيدة: مضاعفات لاجرانج، وشروط كاروش-كون-تاكر (KKT) | Lagrange multipliers, KKT conditions |
| البرمجة التربيعية (ومنها محفظة ماركوفيتز وMOTAD)، والبرمجة القابلة للفصل والكسرية والهندسية والمحدبة | Quadratic, separable, fractional, geometric, convex programming |
| الخوارزميات الاستدلالية والميتا-استدلالية: الخوارزمية الجينية، والتلدين المحاكى، والبحث المحظور، وسرب الجسيمات | Metaheuristics: GA, simulated annealing, tabu search, PSO |

### 9.3 برمجة الأهداف وتعدد الأهداف — Goal & Multi-objective Programming
| النموذج / الطريقة | Model / method |
|---|---|
| برمجة الأهداف الموزونة والتسلسلية (ذات الأولويات) والحد الأدنى-الأعظم | Weighted, preemptive (lexicographic), min-max (Chebyshev) goal programming |
| برمجة الأهداف الضبابية | Fuzzy goal programming |
| البرمجة متعددة الأهداف: المجموع الموزون، وقيد إبسلون، و**رسم حدود باريتو** | Weighted sum, ε-constraint, Pareto frontier |
| التطبيقات: الموازنة بين الربح والمخاطرة والمياه والعمالة في الخطط الزراعية، والتخطيط الإقليمي | Profit–risk–water–labour trade-offs, regional planning |

### 9.4 نماذج اتخاذ القرار متعدد المعايير — MCDM
| النموذج / الطريقة | Model / method |
|---|---|
| العملية الهرمية التحليلية AHP، مع **نسبة الاتساق** ودمج آراء الخبراء (المتوسط الهندسي) | AHP with consistency ratio, group aggregation |
| AHP الضبابي، والشبكية ANP | Fuzzy AHP, ANP |
| TOPSIS، وTOPSIS الضبابي | TOPSIS, fuzzy TOPSIS |
| VIKOR، وPROMETHEE I/II، وELECTRE، وSAW، وWASPAS، وCOPRAS، وMOORA، وEDAS | VIKOR, PROMETHEE, ELECTRE, SAW, WASPAS, COPRAS, MOORA, EDAS |
| أساليب الأوزان: الإنتروبي، وCRITIC، وأفضل-أسوأ (BWM)، وSWARA | Entropy, CRITIC, Best-Worst Method, SWARA weights |
| DEMATEL للعلاقات السببية بين المعايير | DEMATEL |
| **تحليل حساسية الترتيب** لتغير الأوزان، والمقارنة بين نتائج الطرق المختلفة | Rank sensitivity & method comparison |

### 9.5 نماذج النقل والتخصيص — Transportation & Assignment
| النموذج / الطريقة | Model / method |
|---|---|
| الحل الأساسي الأولي: الركن الشمالي الغربي، وأقل تكلفة، و**فوجل التقريبية** | NW corner, least cost, Vogel's approximation |
| اختبار الأمثلية: MODI (u-v) والحجر المتنقل | MODI, stepping stone |
| الحالات الخاصة: غير المتوازنة، والانحلال، والمسارات الممنوعة، وتعظيم الربح | Unbalanced, degeneracy, prohibited routes, maximisation |
| مسائل الشحن العابر (Transshipment) | Transshipment |
| التخصيص: **الطريقة المجرية**، وغير المتوازن، والتعظيم، والقيود | Hungarian method and variants |
| البائع المتجول وتوجيه المركبات (VRP) | TSP, vehicle routing |

### 9.6 نظرية الألعاب ونماذج التفاعل الاستراتيجي — Game Theory & Strategic Interaction
| النموذج / الطريقة | Model / method |
|---|---|
| المباريات الصفرية لشخصين: نقطة السرج، والسيطرة، والاستراتيجيات المختلطة (جبرياً، وبيانياً، وبالبرمجة الخطية) | Two-person zero-sum, saddle point, dominance, mixed strategies |
| المباريات غير الصفرية: **توازن ناش**، ومعضلة السجين، والمباريات المتسلسلة (شجرة اللعبة والاستقراء العكسي) | Nash equilibrium, prisoner's dilemma, sequential games, backward induction |
| نماذج احتكار القلة: كورنو، وبرتراند، وستاكلبرج، والتواطؤ | Cournot, Bertrand, Stackelberg, collusion |
| المباريات التعاونية: **قيمة شابلي**، والنواة، والمساومة لناش (تقاسم المياه والموارد) | Shapley value, core, Nash bargaining |
| المباريات المتكررة والتطورية والمعلومات غير الكاملة (Bayesian) | Repeated, evolutionary, Bayesian games |
| تطبيقات: التفاوض التجاري، وإدارة الموارد المشتركة، وسياسات التسعير | Trade negotiation, common-pool resources, pricing |

### 9.7 نماذج الشبكات وإدارة المشروعات — Networks & Project Management
| النموذج / الطريقة | Model / method |
|---|---|
| أقصر مسار (ديكسترا، فلويد)، وأصغر شجرة ممتدة (بريم، كروسكال)، والتدفق الأقصى (فورد-فولكرسون)، والتدفق الأقل تكلفة | Shortest path, minimum spanning tree, max flow, min-cost flow |
| **CPM**: المسار الحرج، والأوقات المبكرة والمتأخرة، والفائض الكلي والحر، و**مخطط جانت** | Critical path, floats, Gantt chart |
| **PERT**: التقديرات الثلاثة، و**احتمال إنجاز المشروع في موعد محدد** | Three-point estimates, completion probability |
| ضغط المشروع (الوقت-التكلفة)، وتسوية الموارد وتحميلها | Crashing, resource levelling & loading |
| محاكاة مونت كارلو لمدة المشروع | Monte Carlo schedule risk |

### 9.8 نماذج المخزون — Inventory
| النموذج / الطريقة | Model / method |
|---|---|
| الحتمية: EOQ، والإنتاج الاقتصادي EPQ، والعجز المسموح، وخصم الكمية، والأصناف المتعددة بقيود | EOQ, EPQ, planned shortages, quantity discounts, multi-item constrained |
| الاحتمالية: نقطة إعادة الطلب ومخزون الأمان، ونظاما (s, S) و(R, Q)، والمراجعة الدورية | Reorder point & safety stock, (s,S), (R,Q), periodic review |
| نموذج بائع الصحف (الفترة الواحدة) | Newsvendor |
| تحليل ABC وVED | ABC / VED analysis |
| تطبيقات: مخزون الحبوب والأعلاف والأسمدة، والمخزون الاستراتيجي من السلع | Grain, feed, fertiliser & strategic reserves |

### 9.9 نماذج صفوف الانتظار — Queueing
| النموذج / الطريقة | Model / method |
|---|---|
| M/M/1، وM/M/c، وM/M/1/K، وM/M/c/K، والمجتمع المحدود | M/M/1, M/M/c, finite capacity & finite source |
| M/G/1 (بولاتشيك-خينتشين)، وM/D/1، وأولويات الخدمة | M/G/1, M/D/1, priority queues |
| مقاييس الأداء: Lq وL وWq وW وP0 ونسبة الاستخدام | Performance measures |
| **اقتصاديات صفوف الانتظار**: العدد الأمثل لمنافذ الخدمة الذي يقلل التكلفة الكلية | Cost optimisation of servers |
| تطبيقات: صوامع الاستلام، والموانئ، والمستشفيات، والبنوك | Silos, ports, hospitals, banks |

### 9.10 المحاكاة وتحليل القرار — Simulation & Decision Analysis
| النموذج / الطريقة | Model / method |
|---|---|
| توليد الأعداد العشوائية واختبارها، وتوليد المتغيرات من التوزيعات | Random number & variate generation |
| **محاكاة مونت كارلو** مع تحليل المخاطر للمشروعات والأرباح الزراعية | Monte Carlo risk analysis |
| محاكاة الأحداث المتقطعة (صفوف الانتظار والمخزون) | Discrete-event simulation |
| القرار في حالة عدم التأكد: لابلاس، وماكسيمين، وماكسيماكس، وهورويتز، وسافاج (الندم الأدنى) | Laplace, maximin, maximax, Hurwicz, Savage |
| القرار في حالة المخاطرة: القيمة النقدية المتوقعة، والخسارة المتوقعة، و**شجرة القرار**، وقيمة المعلومات الكاملة والعينية (بيز) | EMV, EOL, decision trees, EVPI, EVSI (Bayesian) |
| نظرية المنفعة ودرجة تقبّل المخاطرة (المكافئ المؤكد) | Utility theory, certainty equivalent, risk attitude |
| تطبيقات زراعية للمخاطر: MOTAD وTarget MOTAD، والمتوسط-التباين، وأمان أولاً (Safety-first)، والهيمنة العشوائية | MOTAD, Target MOTAD, E-V, safety-first, stochastic dominance |

### 9.11 النماذج الديناميكية والاحتمالية — Dynamic & Stochastic Models
| النموذج / الطريقة | Model / method |
|---|---|
| **سلاسل ماركوف**: مصفوفة الانتقال، والتنبؤ بالحالات، و**الحالة المستقرة**، والسلاسل الماصّة (متوسط زمن الامتصاص) | Markov chains: transitions, steady state, absorbing chains |
| تطبيقات ماركوف: حصص السوق وولاء المستهلك، وتطور حجم المزارع، وحركة الحيازات | Market shares, brand loyalty, farm-size dynamics |
| عمليات القرار الماركوفية (MDP): تكرار القيمة وتكرار السياسة | Markov decision processes |
| **البرمجة الديناميكية**: الحقيبة، وتوزيع الموارد، وأقصر مسار متعدد المراحل، وجدولة المخزون والإنتاج، واستبدال المعدات | Dynamic programming: knapsack, resource allocation, staged shortest path, production–inventory, equipment replacement |
| البرمجة الديناميكية العشوائية | Stochastic dynamic programming |
| **البرمجة العشوائية**: النموذج ذو المرحلتين مع التعويض، والبرمجة بقيود احتمالية (Chance-constrained)، والمبنية على السيناريوهات، و**الأمثلية المتينة** (Robust optimization) | Two-stage recourse, chance-constrained, scenario-based, robust optimisation |
| تطبيقات: التخطيط الزراعي في ظل عدم تأكد الأمطار والأسعار والمياه | Planning under rainfall, price & water uncertainty |

### 9.12 نماذج تقييم الكفاءة والأداء — Efficiency & Performance
| النموذج / الطريقة | Model / method |
|---|---|
| **تحليل مغلف البيانات DEA**: CCR وBCC (توجيه المدخلات والمخرجات)، وكفاءة الحجم ونوع العائد للسعة | DEA CCR / BCC, input/output orientation, scale efficiency, RTS |
| النماذج المتقدمة: SBM (المعتمد على الفوائض)، والكفاءة الفائقة، والكفاءة التقاطعية، والمخرجات غير المرغوبة (التلوث) | SBM, super-efficiency, cross-efficiency, undesirable outputs |
| كفاءة التكاليف والإيرادات والتخصيصية | Cost, revenue & allocative efficiency |
| DEA عبر الزمن: **مؤشر مالمكويست** وتجزئته، وتحليل النافذة | Malmquist index, window analysis |
| DEA الشبكي (متعدد المراحل) | Network DEA |
| **DEA بالبوتستراب** (Simar-Wilson) لفترات الثقة، ثم تحليل المرحلة الثانية (توبيت، والانحدار المبتور) لمحددات الكفاءة | Bootstrap DEA, second-stage Tobit / truncated regression |
| المقارنة المرجعية: الوحدات المرجعية (Peers) والقيم المستهدفة للتحسين | Benchmarking: peers & improvement targets |
| تحليل الحدود العشوائية SFA (انظر 2.9) والمقارنة بين DEA وSFA | SFA cross-reference & DEA–SFA comparison |
| مؤشرات الأداء المركبة وبطاقة الأداء المتوازن | Composite KPIs, balanced scorecard |

### 9.13 الأنظمة المعقدة — Complex Systems
| النموذج / الطريقة | Model / method |
|---|---|
| **ديناميكيات النظم** (System Dynamics): مخططات الحلقات السببية، ونماذج المخزون والتدفق، والمحاكاة عبر الزمن | Causal loop diagrams, stock-and-flow models, simulation |
| اختبار السياسات بديناميكيات النظم: الأمن الغذائي، والمياه-الغذاء-الطاقة، والنمو السكاني، وسلاسل الإمداد (أثر السوط) | Food security, water–food–energy nexus, population, bullwhip effect |
| تحليل الحساسية واختبار صحة النموذج (الظروف القصوى، وإعادة إنتاج السلوك التاريخي) | Sensitivity & validation tests |
| **المحاكاة القائمة على الوكلاء** (Agent-Based Modelling): وكلاء بقواعد سلوك، وبيئة، وظهور السلوك الجمعي | Agents, rules, environment, emergence |
| تطبيقات الوكلاء: انتشار تبني التكنولوجيا بين المزارعين، وأسواق السلع، واستخدام الأراضي، وانتشار الأوبئة، ونموذج شيلينج | Technology diffusion, commodity markets, land use, epidemics, Schelling |
| تكرارات مونت كارلو لنماذج الوكلاء، وتحليل الحساسية للمعلمات | Monte Carlo replications, parameter sweeps |
| تحليل الشبكات الاجتماعية والاقتصادية (المركزية، والمجتمعات)، والأوتوماتا الخلوية | Network analysis (centrality, communities), cellular automata |

## 10. محور تقييم المشروعات والتحليل المالي — Project Appraisal
- صافي القيمة الحالية، ومعدل العائد الداخلي، ونسبة المنافع إلى التكاليف، وفترة الاسترداد · NPV, IRR, B/C, payback
- تحليل الحساسية والتعادل، والتحليل الاقتصادي بأسعار الظل · sensitivity, switching values, economic analysis
- جداول استهلاك القروض · loan amortization

## 11. محور الأرقام القياسية — Index Numbers
- لاسبير، وباشي، وفيشر، وتورنكفيست، والأرقام المتسلسلة · Laspeyres, Paasche, Fisher, Törnqvist, chain indices
- تحويل السلاسل من الأسعار الجارية إلى الثابتة، وتغيير سنة الأساس · deflating, rebasing

---

## 12. المحاور العرضية — Cross-cutting
| المحور | المحتوى |
|---|---|
| 🏛️ **لوحة صانع القرار** | ملخص بلغة غير فنية، ومحاكاة السيناريوهات، ومؤشرات الإنذار المبكر |
| 📄 **استوديو التقارير** | تجميع النتائج في تقرير، وتصدير Word وPDF وLaTeX، والجداول بنمط APA والمجلات |
| 🤖 **مساعد الذكاء الاصطناعي** | يقترح النموذج، ويكتب الفقرات المنهجية وفقرات النتائج **من الأرقام المحسوبة فعلاً**، ويجيب عن أسئلة الباحث بالعربية |
| 🎓 **الأكاديمية** | مسار تعليمي لكل محور: شرح النظرية، ثم المعادلات، ثم مثال محلول، ثم تمرين على بيانات حقيقية، ثم اختبار وشهادة |
| 🔁 **قابلية إعادة الإنتاج** | تصدير كود Python وR وStata لكل تحليل |
| ✔️ **ضمان الجودة** | مقارنة كل نموذج بمخرجات Stata وR وEViews المنشورة قبل إطلاقه |

---

## 13. ترتيب التنفيذ المقترح — Delivery order
| الإصدار | المحتوى |
|---|---|
| v1.4 | محور الانحدار 1.1 + التشخيص الكامل 1.3 + جدول المجلات + المرونات |
| v1.5 | النماذج الاقتصادية: الإنتاج، والتكاليف، والطلب، والعرض (2.1–2.4) |
| v1.6 | السلاسل الزمنية: جذور الوحدة، وARDL وNARDL، والتكامل المشترك، وVAR وVECM |
| v1.7 | محور التنبؤ الكامل |
| v1.8 | بيانات البانل (الساكن والديناميكي وغير المتجانس) |
| v1.9 | المتغير التابع المحدود + IV والمعادلات الآنية |
| v2.0 | بحوث العمليات (1): البرمجة الخطية والصحيحة، والنقل والتخصيص، والشبكات وPERT/CPM، والمخزون وصفوف الانتظار + تقييم المشروعات |
| v2.0.5 | بحوث العمليات (2): الأهداف وتعدد الأهداف، وMCDM، والكفاءة (DEA وSFA)، والمحاكاة وتحليل القرار |
| v2.0.8 | بحوث العمليات (3): غير الخطية والعددية، ونظرية الألعاب، وماركوف والبرمجة الديناميكية والعشوائية، وديناميكيات النظم والوكلاء |
| v2.1 | الأسواق والتجارة والكلي والتنمية (2.5–2.8) |
| v2.2 | الاستدلال السببي، ومتعدد المتغيرات وSEM، والمكاني |
| v2.3 | استوديو التقارير + مساعد الذكاء الاصطناعي + لوحة صانع القرار |
| v2.4+ | الأكاديمية والمجتمع، والحسابات والاشتراكات |
