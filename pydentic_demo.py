from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field
from typing import Optional



load_dotenv()


model = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",   
    temperature=0
)

# ✅ schema
class Student(BaseModel):
    name: str = "nitish"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description="Student CGPA (0-10)")

structured_model = model.with_structured_output(Student)

prompt = """
Create a student object.
- age must be integer
- email must be valid
"""

result = structured_model.invoke(prompt)

print(result)
print(result.age)
print(result.model_dump())
print(result.model_dump_json())
