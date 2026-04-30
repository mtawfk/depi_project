class CustomTabularModel(MLModel):
    def __init__(self, model_id: str, name: str, description: str, expected_features: List[str], model_path: str):
        super().__init__(
            model_id=model_id,
            name=name,
            description=description,
            input_type="tabular"
        )
        self.expected_features = expected_features
        self.model_path = model_path
        self.model = None
    
    def get_info(self):
        base_info = super().get_info()
        return ModelInfo(
            id=base_info.id,
            name=base_info.name,
            description=base_info.description,
            input_type=base_info.input_type,
            supported_formats=base_info.supported_formats,
            expected_features=self.expected_features
        )
    
    def load(self):
        try:
            # تحميل النموذج بناءً على نوع الملف
            if self.model_path.endswith('.pkl'):
                self.model = joblib.load(self.model_path)
            elif self.model_path.endswith('.h5'):
                self.model = load_keras_model(self.model_path)
            else:
                raise ValueError("Unsupported model format")
            return True
        except Exception as e:
            logging.error(f"Error loading CustomTabularModel: {str(e)}")
            return False
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        try:
            # تحويل الميزات إلى DataFrame
            input_df = pd.DataFrame([features])
            
            # التحقق من وجود جميع الميزات المطلوبة
            missing = [f for f in self.expected_features if f not in input_df.columns]
            if missing:
                raise ValueError(f"Missing features: {missing}")
            
            # إعادة ترتيب الأعمدة حسب الترتيب المتوقع
            X = input_df[self.expected_features]
            
            # التنبؤ
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)
                confidence = float(np.max(proba))
                prediction = int(np.argmax(proba))
            else:
                pred = self.model.predict(X)
                confidence = float(pred[0][0])
                prediction = 1 if confidence > 0.5 else 0
                confidence = max(confidence, 1 - confidence)
            
            return {
                "prediction": str(prediction),
                "confidence": confidence,
                "processing_time": 0.0
            }
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")