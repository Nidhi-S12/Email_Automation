from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class EmailDetails(BaseModel):
    """Model for email details."""
    email_id: Optional[str] = Field(None, description="Email ID")
    subject: Optional[str] = Field(None, description="Email subject")
    sender: Optional[str] = Field(None, description="Email sender")
    body: Optional[str] = Field(None, description="Email body")
    date: Optional[str] = Field(None, description="Email date (YYYY-MM-DD)")
    age_days: Optional[int] = Field(None, description="Age of the email in days from today")
    thread_info: Optional[Dict[str, Any]] = Field(None, description="Thread information")
    is_part_of_thread: Optional[bool] = Field(False, description="Whether this email is part of a thread")
    thread_size: Optional[int] = Field(1, description="Number of emails in this thread")
    thread_position: Optional[int] = Field(1, description="Position of this email in the thread (1 = first)")

    @classmethod
    def from_email_tuple(cls, email_tuple):
        """Create an EmailDetails from an email tuple."""
        if not email_tuple or len(email_tuple) < 5:
            return cls(email_id=None, subject=None)
        
        subject, sender, body, email_id, thread_info = email_tuple
        
        # Extract date from thread_info
        date = ""
        if isinstance(thread_info, dict) and 'date' in thread_info:
            date = thread_info['date']
            
        return cls(
            email_id=email_id,
            subject=subject,
            sender=sender,
            body=body,
            date=date,
            thread_info=thread_info
        )

class EmailResponse(BaseModel):
    """Model for email response information."""
    email_id: str = Field(..., description="Unique identifier for the email")
    subject: str = Field(..., description="Email subject line")
    recipient: str = Field(..., description="Email recipient")
    body: str = Field(default="", description="Draft response body content")
    response_summary: str = Field(..., description="Summary of the response")
    response_needed: bool = Field(..., description="Whether a response was needed")
    draft_saved: bool = Field(default=False, description="Whether a draft was saved")
    created_at: Optional[str] = Field(default=None, description="Timestamp when draft was created")

class EmailResponseList(BaseModel):
    """Wrapper model for a list of email responses."""
    responses: List[EmailResponse] = Field(default_factory=list, description="List of email responses")
