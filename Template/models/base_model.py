from typing import Dict, Any, List, Union
from pydantic import BaseModel

class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    input_type: str
    supported_formats: List[str] = []

class MLModel:
    def __init__(self, model_id: str, name: str, description: str, input_type: str):
        self.model_id = model_id
        self.name = name
        self.description = description
        self.input_type = input_type
        self.model = None
        
    def load(self):
        """تحميل الموديل في الذاكرة"""
        raise NotImplementedError("يجب على الفئات الفرعية تنفيذ load()")
    
    def predict(self, data):
        """إجراء تنبؤ باستخدام الموديل"""
        raise NotImplementedError("يجب على الفئات الفرعية تنفيذ predict()")
    
    def get_info(self) -> ModelInfo:
        """إرجاع معلومات الموديل"""
        return ModelInfo(
            id=self.model_id,
            name=self.name,
            description=self.description,
            input_type=self.input_type
        )