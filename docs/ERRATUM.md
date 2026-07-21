Erratum (2026-07-20)

The submitted thesis (Thesis_UvA_Kaleb_Mazurek.pdf, September 2023) calls the prominence classifier an SVM with count vectorization. That is a naming error. The model actually deployed was Multinomial Naive Bayes with TF-IDF and SMOTE (F1 about 0.73), as its own technical appendix documents; I confirmed it in 2026 from the original code and the saved model. The results are unaffected: every deployed label came from that Naive Bayes model. The "F1 0.79 / 0.65, accuracy about 81%" numbers in the text are from a discarded SVM prototype, not the deployed model.

The submitted PDF is left unchanged as the record; this note is the correction.
