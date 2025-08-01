from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
from gmail_crew_ai.tools.gmail_tools import (
    GetUnreadEmailsTool,
    GetThreadHistoryTool,
    ContextAwareReplyTool,
    get_email_listener,
)
from gmail_crew_ai.crew import GmailCrewAi

# Setup logger
logger = logging.getLogger(__name__)

# Create FastAPI app with metadata
app = FastAPI(
    title="Gmail CrewAI API",
    description="API for email automation with AI-powered assistants",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

class ReplyRequest(BaseModel):
    email_id: str
    body: str
    subject: Optional[str] = None
    include_history: bool = True
    max_history_depth: int = 5
    
    class Config:
        schema_extra = {
            "example": {
                "email_id": "12345",
                "body": "Thank you for your email. I'll review your proposal and get back to you soon.",
                "subject": "Re: Your Project Proposal",
                "include_history": True,
                "max_history_depth": 5
            }
        }

@app.get("/")
def root():
    """
    Gmail CrewAI API - Email automation with AI.
    
    This API provides endpoints to:
    - Fetch unread emails
    - Retrieve email thread history
    - Draft context-aware replies
    - Run the full email automation crew
    
    For complete documentation with examples, visit /docs
    """
    return {
        "name": "Gmail CrewAI API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": [
            {"path": "/emails/unread", "method": "GET", "description": "Fetch unread emails"},
            {"path": "/emails/thread-history", "method": "GET", "description": "Get thread history"},
            {"path": "/emails/draft-reply", "method": "POST", "description": "Draft a reply"},
            {"path": "/crew/run", "method": "POST", "description": "Run full automation"},
            {"path": "/api/listener/*", "method": "GET/POST", "description": "Real-time monitoring control"}
        ],
        "documentation_url": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Gmail CrewAI"
    }

@app.get("/emails/unread")
def get_unread_emails(
    limit: int = Query(5, ge=1, le=50, description="Maximum number of emails to fetch"),
    prioritize_primary: bool = Query(True, description="Whether to prioritize Primary tab emails first")
):
    """
    Fetch unread emails from your Gmail inbox with optional Primary tab prioritization.
    
    Parameters:
    - **limit**: Maximum number of unread emails to retrieve (1-50)  
    - **prioritize_primary**: Whether to prioritize Primary tab emails over Promotions/Social/Updates
    
    Returns a list of unread emails with their details.
    """
    tool = GetUnreadEmailsTool()
    emails = tool._run(limit=limit, prioritize_primary=prioritize_primary)
    return {"emails": emails}

@app.get("/emails/thread-history")
def get_thread_history(
    email_id: str = Query(..., description="Email ID to get thread history for"),
    include_attachments: bool = Query(False, description="Include attachment information"),
    max_depth: int = Query(10, ge=1, le=50, description="Maximum emails to include in history")
):
    """
    Get complete conversation history for an email thread.
    
    Parameters:
    - **email_id**: ID of the email (required)
    - **include_attachments**: Whether to include attachment information
    - **max_depth**: Maximum number of emails to retrieve in the thread
    
    Returns the complete thread history with all messages in chronological order.
    """
    tool = GetThreadHistoryTool()
    history = tool._run(email_id=email_id, include_attachments=include_attachments, max_depth=max_depth)
    return history

@app.post("/emails/draft-reply")
def draft_contextual_reply(request: ReplyRequest):
    """
    Draft a context-aware reply to an email.
    
    The reply will be saved as a draft in your Gmail account. Gmail's threading
    features will ensure it appears correctly in the conversation.
    
    Parameters:
    - **email_id**: ID of the email to reply to
    - **body**: Content of your reply
    - **subject**: Optional subject override
    - **include_history**: Whether to fetch and include conversation history
    - **max_history_depth**: Maximum emails to include in history context
    
    Returns the result of the draft creation operation.
    """
    tool = ContextAwareReplyTool()
    result = tool._run(
        email_id=request.email_id,
        body=request.body,
        subject=request.subject,
        include_history=request.include_history,
        max_history_depth=request.max_history_depth,
    )
    return {"result": result}

@app.post("/crew/run")
def run_crew(email_limit: int = Query(5, ge=1, le=20, description="Number of emails to process")):
    """
    Run the full email processing crew with AI agents.
    
    This endpoint triggers the complete workflow:
    1. Fetches unread emails
    2. Analyzes each email
    3. Determines which emails need responses
    4. Drafts appropriate replies
    
    Parameters:
    - **email_limit**: Number of unread emails to process
    
    Returns the results of the crew execution.
    
    Note: This operation may take some time to complete depending on the
    number of emails being processed.
    """
    result = GmailCrewAi().crew().kickoff(inputs={'email_limit': email_limit})
    return {"result": result}


# -------------------------------
# Real-time Email Monitoring Endpoints
# -------------------------------

@app.post("/api/listener/start")
def start_email_listener():
    """
    Start the real-time email listener.
    
    This endpoint starts an IMAP IDLE listener that monitors your Gmail inbox
    for new emails in real-time. When new emails are detected:
    1. They are automatically fetched and prioritized (Primary tab first)
    2. AI analysis determines which emails need responses
    3. Context-aware draft replies are generated automatically
    4. Results are saved to output files and Gmail drafts
    
    Returns the status of the listener startup operation.
    """
    try:
        listener = get_email_listener()
        success = listener.start_listening()
        
        if success:
            return {
                "status": "started",
                "message": "Real-time email listener started successfully",
                "listener_status": listener.get_status()
            }
        else:
            return {
                "status": "already_running",
                "message": "Email listener is already running",
                "listener_status": listener.get_status()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start email listener: {str(e)}")


@app.post("/api/listener/stop")
def stop_email_listener():
    """
    Stop the real-time email listener.
    
    This endpoint stops the IMAP IDLE listener and terminates real-time
    email monitoring. Any emails currently being processed will complete,
    but no new emails will be detected until the listener is restarted.
    
    Returns the status of the listener shutdown operation.
    """
    try:
        listener = get_email_listener()
        success = listener.stop_listening()
        
        if success:
            return {
                "status": "stopped",
                "message": "Real-time email listener stopped successfully",
                "listener_status": listener.get_status()
            }
        else:
            return {
                "status": "not_running",
                "message": "Email listener was not running",
                "listener_status": listener.get_status()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop email listener: {str(e)}")


@app.get("/api/listener/status")
def get_listener_status():
    """
    Get the current status of the real-time email listener.
    
    Returns detailed information about:
    - Whether the listener is currently active
    - Statistics about emails detected and processed
    - Connection status and error counts
    - Last activity timestamp
    
    This endpoint is useful for monitoring the health and activity
    of the real-time email monitoring system.
    """
    try:
        listener = get_email_listener()
        status = listener.get_status()
        
        return {
            "listener_status": status,
            "message": "Listener status retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get listener status: {str(e)}")


@app.post("/api/listener/restart")
def restart_email_listener():
    """
    Restart the real-time email listener.
    
    This endpoint stops the current listener (if running) and starts a new one.
    Useful for recovering from connection issues or applying configuration changes.
    
    Returns the status of the restart operation.
    """
    try:
        listener = get_email_listener()
        
        # Stop if running
        listener.stop_listening()
        
        # Start again
        success = listener.start_listening()
        
        if success:
            return {
                "status": "restarted",
                "message": "Real-time email listener restarted successfully",
                "listener_status": listener.get_status()
            }
        else:
            return {
                "status": "failed",
                "message": "Failed to restart email listener",
                "listener_status": listener.get_status()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart email listener: {str(e)}")

@app.delete("/api/listener/clear-history")
def clear_processing_history():
    """
    Clear the processing history for both processed emails and generated responses.
    
    This endpoint clears the in-memory history of processed emails and generated responses
    from the real-time email listener. Useful for cleanup or testing purposes.
    
    Returns:
        Status of the clear operation and counts of cleared items
    """
    try:
        listener = get_email_listener()
        
        # Get processor instance and clear history
        if hasattr(listener, 'email_processor') and listener.email_processor:
            # Get counts before clearing
            processed_count = len(listener.email_processor.get_processed_emails())
            responses_count = len(listener.email_processor.get_generated_responses())
            
            # Clear the history
            listener.email_processor.clear_history()
            
            return {
                "status": "success",
                "message": "Processing history cleared successfully",
                "cleared_emails": processed_count,
                "cleared_responses": responses_count
            }
        else:
            return {
                "status": "no_processor",
                "message": "Email listener processor not available",
                "cleared_emails": 0,
                "cleared_responses": 0
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear processing history: {str(e)}")


@app.get("/api/listener/processed-emails")
def get_processed_emails(limit: Optional[int] = Query(default=50, ge=1, le=500)):
    """
    Get all processed emails from the real-time monitoring.
    
    Args:
        limit: Maximum number of processed emails to return (default: 50, max: 500)
    
    Returns a list of emails that have been detected and processed
    by the real-time email monitoring system, including:
    - Email metadata (subject, sender, timestamp)
    - Processing status and results
    - Any errors encountered during processing
    
    This endpoint is useful for reviewing the history of
    processed emails and monitoring system activity.
    """
    try:
        listener = get_email_listener()
        if hasattr(listener, 'email_processor') and listener.email_processor:
            all_processed_emails = listener.email_processor.get_processed_emails()
            # Apply limit and get most recent emails first
            limited_emails = all_processed_emails[-limit:] if len(all_processed_emails) > limit else all_processed_emails
            limited_emails.reverse()  # Most recent first
            
            return {
                "processed_emails": limited_emails,
                "total_count": len(all_processed_emails),
                "returned_count": len(limited_emails),
                "status": "success",
                "message": f"Retrieved {len(limited_emails)} most recent processed emails"
            }
        else:
            return {
                "processed_emails": [],
                "total_count": 0,
                "returned_count": 0,
                "status": "no_processor",
                "message": "Email listener processor not available"
            }
    except Exception as e:
        logger.error(f"Error getting processed emails: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get processed emails: {str(e)}")


@app.get("/api/listener/generated-responses")
def get_generated_responses(limit: Optional[int] = Query(default=50, ge=1, le=500)):
    """
    Get all generated responses from the real-time monitoring.
    
    Args:
        limit: Maximum number of generated responses to return (default: 50, max: 500)
    
    Returns a list of draft responses that have been generated
    by the CrewAI system for processed emails, including:
    - Response content and metadata
    - Associated email information
    - Generation timestamp and status
    
    This endpoint is useful for reviewing generated responses
    and monitoring the AI response generation system.
    """
    try:
        listener = get_email_listener()
        if hasattr(listener, 'email_processor') and listener.email_processor:
            all_generated_responses = listener.email_processor.get_generated_responses()
            # Apply limit and get most recent responses first
            limited_responses = all_generated_responses[-limit:] if len(all_generated_responses) > limit else all_generated_responses
            limited_responses.reverse()  # Most recent first
            
            return {
                "generated_responses": limited_responses,
                "total_count": len(all_generated_responses),
                "returned_count": len(limited_responses),
                "status": "success",
                "message": f"Retrieved {len(limited_responses)} most recent generated responses"
            }
        else:
            return {
                "generated_responses": [],
                "total_count": 0,
                "returned_count": 0,
                "status": "no_processor",
                "message": "Email listener processor not available"
            }
    except Exception as e:
        logger.error(f"Error getting generated responses: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get generated responses: {str(e)}")


@app.get("/api/listener/activity-logs")
def get_activity_logs(limit: Optional[int] = Query(default=50, ge=1, le=200)):
    """
    Get recent activity logs from the real-time email monitoring.
    
    Args:
        limit: Maximum number of log entries to return (default: 50, max: 200)
    
    Returns the most recent activity logs from the email listener,
    including timestamps, events, and status changes.
    """
    try:
        listener = get_email_listener()
        if hasattr(listener, 'activity_logs') and listener.activity_logs:
            # Get most recent logs first
            recent_logs = listener.activity_logs[-limit:] if len(listener.activity_logs) > limit else listener.activity_logs
            recent_logs.reverse()  # Most recent first
            
            return {
                "activity_logs": recent_logs,
                "total_count": len(listener.activity_logs),
                "returned_count": len(recent_logs),
                "status": "success"
            }
        else:
            return {
                "activity_logs": [],
                "total_count": 0,
                "returned_count": 0,
                "status": "no_logs",
                "message": "No activity logs available"
            }
    except Exception as e:
        logger.error(f"Error getting activity logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get activity logs: {str(e)}")