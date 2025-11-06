from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path
from config import BASE_DIR

DATABASE_URL = f"sqlite:///{BASE_DIR / 'storage' / 'data' / 'app.db'}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    size = Column(Integer, nullable=False)
    hash = Column(String, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.now)
    status = Column(String, default="indexed")

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)

def init_db():
    """Khởi tạo database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency để get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

