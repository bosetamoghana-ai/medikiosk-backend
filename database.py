# database.py
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./medikiosk.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ConsultationRecord(Base):
    __tablename__ = "consultation_records"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    session_id = Column(String, unique=True, index=True)
    department = Column(String, default="Allopathy")
    raw_transcript = Column(Text)
    clinical_summary = Column(Text)
    status = Column(String, default="pending_review")
    doctor_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()