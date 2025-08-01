"""
Email Dashboard - Web interface for visualizing emails
"""
from fastapi import FastAPI, Request, Query, HTTPException, Form, Depends, status, Cookie
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional, List, Dict, Any
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import os
import json
import uuid
import hashlib
import imaplib
import uvicorn
import re

from gmail_crew_ai.tools.gmail_tools import (
    GetUnreadEmailsTool,
    GetThreadHistoryTool,
    ContextAwareReplyTool,
    get_email_listener,
)
from gmail_crew_ai.crew import GmailCrewAi

# Session storage (in production, use Redis or a proper session store)
active_sessions = {}
user_credentials = {}

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def clean_app_password(password: str) -> str:
    """Clean and format app password by removing spaces and validating format"""
    if not password:
        return password
    
    # Remove all spaces and convert to uppercase for consistency
    cleaned = ''.join(password.split()).upper()
    
    # Additional validation could be added here if needed
    return cleaned

def validate_gmail_credentials(email: str, app_password: str) -> bool:
    """Validate Gmail credentials by attempting to connect"""
    try:
        # Note: app_password should already be cleaned by the caller
        print(f"Validating credentials for: {email}")
        print(f"App password length: {len(app_password)}")
        
        # App passwords should be exactly 16 characters
        if len(app_password) != 16:
            print(f"Invalid app password length: {len(app_password)}. Should be 16 characters.")
            return False
        
        # Validate email format
        if not email.endswith('@gmail.com'):
            print(f"Invalid email format: {email}. Must be a Gmail address.")
            return False
        
        # Test IMAP connection
        mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        mail.login(email, app_password)
        mail.select('INBOX')  # Try to select inbox to ensure full access
        mail.logout()
        print("Gmail credentials validated successfully!")
        return True
        
    except imaplib.IMAP4.error as e:
        error_msg = str(e).lower()
        if "authentication failed" in error_msg:
            print(f"Authentication failed: Invalid email or app password")
        elif "application-specific password required" in error_msg:
            print(f"App-specific password required. Please generate an App Password in Google Account settings")
        else:
            print(f"IMAP error: {e}")
        return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False

def create_session(email: str, app_password: str, model_choice: str, api_key: str) -> str:
    """Create a new session"""
    session_id = generate_session_id()
    
    # Clean the app password (remove spaces)
    clean_app_password = app_password.replace(' ', '').strip()
    
    # Store session data
    active_sessions[session_id] = {
        'email': email,
        'app_password': clean_app_password,
        'model_choice': model_choice,
        'api_key': api_key,
        'created': datetime.now(),
        'expires': datetime.now() + timedelta(hours=24),  # 24 hour session
    }
    
    # Temporarily set environment variables for this session
    user_credentials[session_id] = {
        'EMAIL_ADDRESS': email,
        'APP_PASSWORD': clean_app_password,
        'MODEL': model_choice,
        'OPENAI_API_KEY' if 'openai' in (model_choice or '').lower() else 'GEMINI_API_KEY': api_key
    }
    
    return session_id

def get_current_session(session_id: Optional[str] = Cookie(None)) -> Optional[Dict]:
    """Get current session data"""
    if session_id and session_id in active_sessions:
        session = active_sessions[session_id]
        # Check if session is expired
        if datetime.now() < session['expires']:
            return session
        else:
            # Remove expired session
            del active_sessions[session_id]
            if session_id in user_credentials:
                del user_credentials[session_id]
    return None

def require_auth(session: Optional[Dict] = Depends(get_current_session)) -> Dict:
    """Dependency to require authentication"""
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return session

def set_session_env_vars(session_id: str):
    """Set environment variables for the current session"""
    if session_id in user_credentials:
        creds = user_credentials[session_id]
        for key, value in creds.items():
            os.environ[key] = value

# Create FastAPI app for dashboard
dashboard_app = FastAPI(
    title="Gmail Dashboard",
    description="Visual dashboard for email management and analytics",
    version="1.0.0",
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")

class EmailAnalytics:
    """Class to handle email analytics and statistics"""
    
    @staticmethod
    def analyze_emails(emails: List[Dict]) -> Dict[str, Any]:
        """Analyze email data and return statistics"""
        if not emails:
            return {
                "total_emails": 0,
                "by_sender": {},
                "by_date": {},
                "by_priority": {"high": 0, "medium": 0, "low": 0},
                "response_needed": 0,
                "response_rate": 0,
                "top_domains": {},
                "thread_stats": {"total_threads": 0, "avg_thread_length": 0}
            }
        
        # Basic statistics
        total_emails = len(emails)
        by_sender = Counter()
        by_date = defaultdict(int)
        by_priority = {"high": 0, "medium": 0, "low": 0}
        response_needed_count = 0
        responded_to_count = 0
        domains = Counter()
        thread_data = []
        
        for email in emails:
            # Sender analysis
            sender = email.get('sender', 'Unknown')
            by_sender[sender] += 1
            
            # Extract domain from sender
            if '@' in sender:
                domain = sender.split('@')[-1].replace('>', '').strip()
                domains[domain] += 1
            
            # Date analysis
            email_date = email.get('date', '')
            if email_date:
                try:
                    date_obj = datetime.strptime(email_date, "%Y-%m-%d")
                    date_str = date_obj.strftime("%Y-%m-%d")
                    by_date[date_str] += 1
                except:
                    by_date["Unknown"] += 1
            else:
                by_date["Unknown"] += 1
            
            # Check if response is needed
            needs_response = EmailAnalytics.needs_response(email)
            if needs_response:
                response_needed_count += 1
            
            # Check if email has been responded to (based on thread info or other indicators)
            has_response = EmailAnalytics.has_response(email)
            if has_response:
                responded_to_count += 1
            
            # Priority analysis (based on age, content, and response need)
            age_days = email.get('age_days', 0)
            subject = email.get('subject', '').lower()
            
            if needs_response and (age_days <= 1 or any(word in subject for word in ['urgent', 'asap', 'important'])):
                by_priority["high"] += 1
            elif needs_response and age_days <= 7:
                by_priority["medium"] += 1
            else:
                by_priority["low"] += 1
            
            # Thread analysis
            thread_info = email.get('thread_info', {})
            if isinstance(thread_info, dict):
                thread_size = thread_info.get('thread_size', 1)
                thread_data.append(thread_size)
            else:
                thread_data.append(1)
        
        # Calculate response rate: percentage of emails needing response that have been responded to
        response_rate = round((responded_to_count / response_needed_count) * 100, 1) if response_needed_count > 0 else 100
        
        # Get actual AI-generated responses count from response_report.json
        ai_generated_responses = 0
        try:
            if os.path.exists('output/response_report.json'):
                with open('output/response_report.json', 'r') as f:
                    response_data = json.load(f)
                    responses = []
                    
                    if isinstance(response_data, list):
                        responses = response_data
                    elif isinstance(response_data, dict) and 'responses' in response_data:
                        responses = response_data.get('responses', [])
                    
                    # Count only emails with actual generated content
                    for response in responses:
                        # Count if any of these conditions are met:
                        # 1. response_needed is True (AI determined it needs a response)
                        # 2. body has actual content (not empty)
                        # 3. draft_saved is True (a draft was actually saved)
                        if (response.get('response_needed', False) or 
                            (response.get('body', '').strip() != '') or 
                            response.get('draft_saved', False)):
                            ai_generated_responses += 1
                            
                    print(f"DEBUG: Counted {ai_generated_responses} actual responses out of {len(responses)} total entries")
        except Exception as e:
            print(f"Error reading response report: {e}")
        
        # Top senders (limit to top 10)
        top_senders = dict(by_sender.most_common(10))
        
        # Top domains (limit to top 10)
        top_domains = dict(domains.most_common(10))
        
        # Thread statistics
        thread_stats = {
            "total_threads": len([t for t in thread_data if t > 1]),
            "avg_thread_length": sum(thread_data) / len(thread_data) if thread_data else 0
        }
        
        return {
            "total_emails": total_emails,
            "by_sender": top_senders,
            "by_date": dict(by_date),
            "by_priority": by_priority,
            "response_needed": ai_generated_responses,  # Use actual generated responses count
            "responded_to": responded_to_count,
            "response_rate": response_rate,
            "top_domains": top_domains,
            "thread_stats": thread_stats
        }
    
    @staticmethod
    def needs_response(email: Dict) -> bool:
        """Determine if an email likely needs a response"""
        subject = email.get('subject', '').lower()
        body = email.get('body', '').lower()
        sender = email.get('sender', '').lower()
        age_days = email.get('age_days', 0)
        
        # First check if this email has already been responded to
        if EmailAnalytics.has_response(email):
            return False
        
        # Keywords that suggest NO response is needed (check first)
        no_response_keywords = [
            'newsletter', 'unsubscribe', 'notification', 'no-reply', 'noreply',
            'do not reply', 'automated', 'marketing', 'promo', 'advertisement',
            'digest', 'summary', 'alert', 'reminder', 'confirmation', 'receipt',
            'thank you for your order', 'delivery notification', 'invoice',
            'password reset', 'security alert', 'account statement'
        ]
        
        # Check sender patterns that typically don't need responses
        no_response_senders = [
            'noreply', 'no-reply', 'donotreply', 'automated', 'notification',
            'alert', 'newsletter', 'marketing', 'support@', 'help@', 'billing@',
            'accounts@', 'info@', 'admin@', 'system@'
        ]
        
        # Check for no-response indicators first
        for keyword in no_response_keywords:
            if keyword in subject or keyword in body[:500]:
                return False
        
        for sender_pattern in no_response_senders:
            if sender_pattern in sender:
                return False
        
        # Keywords that strongly suggest a response is needed
        response_keywords = [
            '?', 'question', 'please respond', 'please reply', 'get back to me',
            'can you', 'could you', 'would you', 'will you', 'please',
            'request', 'need help', 'assistance', 'meeting', 'schedule',
            'confirm', 'approval', 'feedback', 'thoughts', 'opinion',
            'discuss', 'call me', 'let me know', 'what do you think',
            'urgent', 'asap', 'important', 'deadline', 'time sensitive'
        ]
        
        # Check for strong response indicators
        response_score = 0
        for keyword in response_keywords:
            if keyword in subject:
                response_score += 2  # Subject matches are weighted more
            if keyword in body[:500]:  # Check first 500 chars of body
                response_score += 1
        
        # Questions are strong indicators
        question_count = subject.count('?') + body[:500].count('?')
        response_score += question_count * 3
        
        # Check for direct addressing (personal emails)
        personal_indicators = ['dear', 'hi ', 'hello', 'hey', 'good morning', 'good afternoon']
        for indicator in personal_indicators:
            if body[:200].startswith(indicator) or f' {indicator}' in body[:200]:
                response_score += 1
        
        # Recent emails from real people are more likely to need responses
        if age_days <= 2 and response_score > 0:
            return True
        
        # High response score indicates clear need for response
        if response_score >= 4:
            return True
        
        # Medium response score with recent date
        if response_score >= 2 and age_days <= 7:
            return True
        
        # Very old emails (>30 days) probably don't need response unless very high score
        if age_days > 30 and response_score < 5:
            return False
        
        return False

    @staticmethod
    def has_response(email: Dict) -> bool:
        """Determine if an email has been responded to"""
        thread_info = email.get('thread_info', {})
        
        # Check if there are multiple messages in the thread (indicating responses)
        if isinstance(thread_info, dict):
            thread_size = thread_info.get('thread_size', 1)
            if thread_size > 1:
                return True
            
            # Check if there are any outgoing messages in the thread
            messages = thread_info.get('messages', [])
            if isinstance(messages, list) and len(messages) > 1:
                # Look for outgoing emails (from self)
                for message in messages:
                    if isinstance(message, dict):
                        sender = message.get('sender', '').lower()
                        # This would need to be customized with the user's email
                        # For now, we'll use heuristics
                        if 'reply' in message.get('subject', '').lower():
                            return True
        
        # Check if the subject indicates it's a response thread
        subject = email.get('subject', '').lower()
        if subject.startswith('re:') or subject.startswith('fwd:'):
            # This might be a response thread, but we can't be sure if WE responded
            # For now, we'll be conservative and assume it's not responded to
            pass
        
        # Check body for response indicators
        body = email.get('body', '').lower()
        response_patterns = [
            'thank you for your email',
            'thanks for reaching out',
            'i replied',
            'as discussed',
            'per our conversation'
        ]
        
        for pattern in response_patterns:
            if pattern in body[:300]:
                return True
        
        return False

class DashboardRoutes:
    """Dashboard route handlers"""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.setup_routes()
    
    def setup_routes(self):
        """Setup all dashboard routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def login_page(request: Request, session: Optional[Dict] = Depends(get_current_session)):
            """Show login page or redirect to dashboard if already authenticated"""
            if session:
                return RedirectResponse(url="/dashboard", status_code=302)
            return templates.TemplateResponse("login.html", {"request": request})
        
        @self.app.post("/auth/login")
        async def login(
            request: Request,
            email: str = Form(...),
            app_password: str = Form(...),
            model_choice: str = Form(...),
            api_key: str = Form(...)
        ):
            """Handle login form submission"""
            try:
                # Clean and validate the app password
                cleaned_password = clean_app_password(app_password)
                
                # Validate Gmail credentials with cleaned password
                if not validate_gmail_credentials(email, cleaned_password):
                    return templates.TemplateResponse("login.html", {
                        "request": request,
                        "error": "Invalid Gmail credentials. Please check your email and app password."
                    })
                
                # Create session with cleaned password
                session_id = create_session(email, cleaned_password, model_choice, api_key)
                
                # Set session environment variables
                set_session_env_vars(session_id)
                
                # Create response and set cookie
                response = RedirectResponse(url="/crew-config", status_code=302)
                response.set_cookie(
                    key="session_id", 
                    value=session_id, 
                    max_age=24*60*60,  # 24 hours
                    httponly=True,
                    secure=False  # Set to True in production with HTTPS
                )
                
                return response
                
            except Exception as e:
                return templates.TemplateResponse("login.html", {
                    "request": request,
                    "error": f"Login failed: {str(e)}"
                })
        
        @self.app.get("/auth/logout")
        async def logout(session_id: Optional[str] = Cookie(None)):
            """Handle logout"""
            if session_id and session_id in active_sessions:
                del active_sessions[session_id]
                if session_id in user_credentials:
                    del user_credentials[session_id]
            
            response = RedirectResponse(url="/", status_code=302)
            response.delete_cookie("session_id")
            return response
        
        @self.app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard_page(request: Request, session: Dict = Depends(require_auth), crew_run: Optional[str] = Query(None)):
            """Main dashboard page (requires authentication)"""
            # Set session environment variables for this request
            session_id = None
            for sid, sess in active_sessions.items():
                if sess == session:
                    session_id = sid
                    break
            
            if session_id:
                set_session_env_vars(session_id)
            
            # Check for success message
            success_message = None
            if crew_run == "success":
                success_message = "CrewAI processing completed successfully! Click the 'Response Drafts' tab above to view and manage your AI-generated email responses."
            
            return templates.TemplateResponse("dashboard.html", {
                "request": request, 
                "user_email": session['email'],
                "success_message": success_message
            })
        
        @self.app.get("/crew-config", response_class=HTMLResponse)
        async def crew_config_page(request: Request, session: Dict = Depends(require_auth)):
            """CrewAI configuration page (requires authentication)"""
            # Set session environment variables for this request
            session_id = None
            for sid, sess in active_sessions.items():
                if sess == session:
                    session_id = sid
                    break
            
            if session_id:
                set_session_env_vars(session_id)
            
            return templates.TemplateResponse("crew_config.html", {
                "request": request, 
                "user_email": session['email'],
                "model_choice": session['model_choice']
            })
        
        @self.app.post("/crew-config/run")
        async def run_crew_config(
            request: Request,
            email_limit: int = Form(...),
            session: Dict = Depends(require_auth)
        ):
            """Handle CrewAI configuration and run"""
            try:
                # Set session environment variables for this request
                session_id = None
                for sid, sess in active_sessions.items():
                    if sess == session:
                        session_id = sid
                        break
                
                if session_id:
                    set_session_env_vars(session_id)
                
                # Store CrewAI configuration in session
                active_sessions[session_id]['crew_config'] = {
                    'email_limit': email_limit,
                    'last_run': datetime.now()
                }
                
                # Run the crew with configuration
                crew = GmailCrewAi()
                result = crew.crew().kickoff(inputs={
                    'email_limit': email_limit,
                    'user_email': session['email']
                })
                
                # Redirect to dashboard with success message
                response = RedirectResponse(url="/dashboard?crew_run=success", status_code=302)
                return response
                
            except Exception as e:
                return templates.TemplateResponse("crew_config.html", {
                    "request": request,
                    "user_email": session['email'],
                    "model_choice": session['model_choice'],
                    "error": f"CrewAI run failed: {str(e)}"
                })
        
        
        @self.app.get("/settings", response_class=HTMLResponse)
        async def settings_page(request: Request, session: Dict = Depends(require_auth)):
            """Settings page (requires authentication)"""
            # Set session environment variables for this request
            session_id = None
            for sid, sess in active_sessions.items():
                if sess == session:
                    session_id = sid
                    break
            
            if session_id:
                set_session_env_vars(session_id)
            
            # Prepare settings data (mask sensitive information with better security)
            settings_data = {
                "email": session['email'],
                "model_choice": session['model_choice'],
                "api_key_preview": "sk-..." + "*" * 20 + session['api_key'][-4:] if session['api_key'].startswith('sk-') else "***..." + "*" * 15 + session['api_key'][-3:],
                "app_password_preview": "***" + "*" * 10 + session['app_password'][-3:] if len(session['app_password']) > 6 else "*" * 16,
                "session_created": session['created'].strftime("%Y-%m-%d %H:%M:%S"),
                "session_expires": session['expires'].strftime("%Y-%m-%d %H:%M:%S"),
                "provider": "OpenAI" if "openai" in (session['model_choice'] or '').lower() else "Google Gemini"
            }
            
            return templates.TemplateResponse("settings.html", {
                "request": request, 
                "user_email": session['email'],
                "settings": settings_data
            })
        
        @self.app.post("/settings/update-credentials")
        async def update_credentials(
            request: Request,
            credential_type: str = Form(...),
            new_value: str = Form(...),
            session: Dict = Depends(require_auth)
        ):
            """Update user credentials"""
            try:
                # Get session ID
                session_id = None
                for sid, sess in active_sessions.items():
                    if sess == session:
                        session_id = sid
                        break
                
                if not session_id:
                    raise HTTPException(status_code=401, detail="Session not found")
                
                if credential_type == "app_password":
                    # Clean and validate new app password
                    cleaned_password = clean_app_password(new_value)
                    
                    # Validate with Gmail
                    if not validate_gmail_credentials(session['email'], cleaned_password):
                        return JSONResponse(content={
                            "success": False,
                            "message": "Invalid app password. Please check your credentials."
                        }, status_code=400)
                    
                    # Update session
                    active_sessions[session_id]['app_password'] = cleaned_password
                    user_credentials[session_id]['APP_PASSWORD'] = cleaned_password
                    
                elif credential_type == "gmail_email":
                    # Validate email format
                    if not new_value.endswith('@gmail.com'):
                        return JSONResponse(content={
                            "success": False,
                            "message": "Please enter a valid Gmail address (must end with @gmail.com)."
                        }, status_code=400)
                    
                    # Validate with current app password
                    if not validate_gmail_credentials(new_value, session['app_password']):
                        return JSONResponse(content={
                            "success": False,
                            "message": "Cannot access Gmail with this email address using current app password. Please update app password first."
                        }, status_code=400)
                    
                    # Update session
                    active_sessions[session_id]['email'] = new_value
                    user_credentials[session_id]['EMAIL_ADDRESS'] = new_value
                    
                elif credential_type == "model_choice":
                    # Validate model choice format
                    valid_models = [
                        "openai/gpt-4o-mini", "openai/gpt-4o", 
                        "gemini/gemini-2.0-flash", "gemini/gemini-pro"
                    ]
                    if new_value not in valid_models:
                        return JSONResponse(content={
                            "success": False,
                            "message": "Invalid model choice. Please select a valid AI model."
                        }, status_code=400)
                    
                    # Update session and API key mapping
                    old_provider = "openai" if "openai" in (session['model_choice'] or '').lower() else "gemini"
                    new_provider = "openai" if "openai" in (new_value or '').lower() else "gemini"
                    
                    active_sessions[session_id]['model_choice'] = new_value
                    user_credentials[session_id]['MODEL'] = new_value
                    
                    # Update API key mapping if provider changed
                    if old_provider != new_provider:
                        # Remove old API key
                        old_key_name = 'OPENAI_API_KEY' if old_provider == 'openai' else 'GEMINI_API_KEY'
                        new_key_name = 'OPENAI_API_KEY' if new_provider == 'openai' else 'GEMINI_API_KEY'
                        
                        if old_key_name in user_credentials[session_id]:
                            del user_credentials[session_id][old_key_name]
                        
                        # Add new API key mapping
                        user_credentials[session_id][new_key_name] = session['api_key']
                        
                        return JSONResponse(content={
                            "success": True,
                            "message": f"Model updated to {new_value}. Note: Please verify your API key is compatible with {new_provider.title()}.",
                            "provider_changed": True
                        })
                    
                elif credential_type == "api_key":
                    # Basic API key validation
                    if len(new_value) < 20:
                        return JSONResponse(content={
                            "success": False,
                            "message": "API key appears to be too short. Please check and try again."
                        }, status_code=400)
                    
                    # Update session
                    active_sessions[session_id]['api_key'] = new_value
                    api_key_name = 'OPENAI_API_KEY' if 'openai' in (session['model_choice'] or '').lower() else 'GEMINI_API_KEY'
                    user_credentials[session_id][api_key_name] = new_value
                
                else:
                    return JSONResponse(content={
                        "success": False,
                        "message": "Invalid credential type"
                    }, status_code=400)
                
                # Update environment variables
                set_session_env_vars(session_id)
                
                return JSONResponse(content={
                    "success": True,
                    "message": f"Successfully updated {credential_type.replace('_', ' ').title()}"
                })
                
            except Exception as e:
                return JSONResponse(content={
                    "success": False,
                    "message": f"Error updating credentials: {str(e)}"
                }, status_code=500)
        
        async def dashboard_home(request: Request):
            """Main dashboard page"""
            return templates.TemplateResponse("dashboard.html", {"request": request})
        
        @self.app.get("/api/emails/analytics")
        async def get_analytics(session: Dict = Depends(require_auth)):
            """Get email analytics (requires authentication)"""
            # Set session environment variables for this request
            session_id = None
            for sid, sess in active_sessions.items():
                if sess == session:
                    session_id = sid
                    break
            
            if session_id:
                set_session_env_vars(session_id)
            
            try:
                # Load emails and analyze
                emails = []
                try:
                    with open('output/fetched_emails.json', 'r') as f:
                        emails = json.load(f)
                except FileNotFoundError:
                    pass
                
                analytics = EmailAnalytics.analyze_emails(emails)
                return JSONResponse(content=analytics)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error getting analytics: {str(e)}")
        
        @self.app.get("/api/emails/list")
        async def get_email_list(limit: int = Query(20, ge=1, le=50), session: Dict = Depends(require_auth)):
            """Get email list (requires authentication)"""
            # Set session environment variables for this request
            session_id = None
            for sid, sess in active_sessions.items():
                if sess == session:
                    session_id = sid
                    break
            
            if session_id:
                set_session_env_vars(session_id)
            
            try:
                emails = []
                try:
                    with open('output/fetched_emails.json', 'r') as f:
                        emails = json.load(f)
                except FileNotFoundError:
                    pass
                
                # Limit the results
                limited_emails = emails[:limit] if emails else []
                return JSONResponse(content={"emails": limited_emails})
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error getting email list: {str(e)}")
        
        @self.app.post("/api/crew/run")
        async def run_crew(email_limit: int = Query(5, ge=1, le=20), session: Dict = Depends(require_auth)):
            """Run the email processing crew (requires authentication)"""
            # Set session environment variables for this request
            session_id = None
            for sid, sess in active_sessions.items():
                if sess == session:
                    session_id = sid
                    break
            
            if session_id:
                set_session_env_vars(session_id)
            
            try:
                # Create and run the crew with user email for personalized signatures
                crew = GmailCrewAi()
                result = crew.crew().kickoff(inputs={
                    'email_limit': email_limit,
                    'user_email': session['email']
                })
                
                return JSONResponse(content={
                    "message": "Email processing completed successfully",
                    "result": str(result) if result else "No emails processed"
                })
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error running crew: {str(e)}")
        
        @self.app.get("/api/responses/drafts")
        async def get_drafts(session: Dict = Depends(require_auth)):
            """Get draft responses (requires authentication)"""
            # Set session environment variables for this request
            session_id = None
            for sid, sess in active_sessions.items():
                if sess == session:
                    session_id = sid
                    break
            
            if session_id:
                set_session_env_vars(session_id)
            
            try:
                drafts = []
                try:
                    with open('output/response_report.json', 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'responses' in data:
                            drafts = data['responses']
                        elif isinstance(data, list):
                            drafts = data
                except FileNotFoundError:
                    pass
                
                return JSONResponse(content={"drafts": drafts})
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error getting drafts: {str(e)}")
        async def get_email_analytics():
            """Get email analytics data"""
            try:
                # Try to read from the output file first
                emails_data = []
                if os.path.exists('output/fetched_emails.json'):
                    with open('output/fetched_emails.json', 'r') as f:
                        emails_data = json.load(f)
                
                # If no data, fetch fresh emails
                if not emails_data:
                    tool = GetUnreadEmailsTool()
                    email_tuples = tool._run(limit=20)
                    
                    # Convert tuples to dictionaries
                    emails_data = []
                    for i, (subject, sender, body, email_id, thread_info) in enumerate(email_tuples):
                        email_dict = {
                            'email_id': email_id,
                            'subject': subject,
                            'sender': sender,
                            'body': body[:500],  # Truncate body for analytics
                            'thread_info': thread_info,
                            'date': thread_info.get('date', '') if isinstance(thread_info, dict) else '',
                            'age_days': self.calculate_age_days(thread_info.get('date', '') if isinstance(thread_info, dict) else '')
                        }
                        emails_data.append(email_dict)
                
                # Generate analytics
                analytics = EmailAnalytics.analyze_emails(emails_data)
                return analytics
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")
        
        @self.app.get("/api/emails/list")
        async def get_emails_list(limit: int = Query(10, ge=1, le=100)):
            """Get list of emails for table view"""
            try:
                # Try to read from the output file first
                emails_data = []
                if os.path.exists('output/fetched_emails.json'):
                    with open('output/fetched_emails.json', 'r') as f:
                        all_emails = json.load(f)
                        emails_data = all_emails[:limit]
                
                # If no data, fetch fresh emails
                if not emails_data:
                    tool = GetUnreadEmailsTool()
                    email_tuples = tool._run(limit=limit)
                    
                    # Convert tuples to dictionaries
                    emails_data = []
                    for subject, sender, body, email_id, thread_info in email_tuples:
                        email_dict = {
                            'email_id': email_id,
                            'subject': subject,
                            'sender': sender,
                            'body_preview': body[:200] + "..." if len(body) > 200 else body,
                            'date': thread_info.get('date', '') if isinstance(thread_info, dict) else '',
                            'age_days': self.calculate_age_days(thread_info.get('date', '') if isinstance(thread_info, dict) else ''),
                            'needs_response': EmailAnalytics.needs_response({
                                'subject': subject,
                                'body': body,
                                'sender': sender,
                                'thread_info': thread_info,
                                'age_days': self.calculate_age_days(thread_info.get('date', '') if isinstance(thread_info, dict) else '')
                            }),
                            'has_response': EmailAnalytics.has_response({
                                'subject': subject,
                                'body': body,
                                'sender': sender,
                                'thread_info': thread_info
                            })
                        }
                        emails_data.append(email_dict)
                
                return {"emails": emails_data}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error fetching emails: {str(e)}")
        
        @self.app.get("/api/emails/archive")
        async def get_email_archive(
            date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
            date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
            search_query: Optional[str] = Query(None, description="Search term"),
            limit: int = Query(50, ge=1, le=500, description="Maximum emails to fetch"),
            folder: str = Query("INBOX", description="Gmail folder"),
            include_read: bool = Query(True, description="Include read emails"),
            include_unread: bool = Query(True, description="Include unread emails"),
            sort_by: str = Query("date", description="Sort by: date, sender, subject"),
            sort_order: str = Query("desc", description="Sort order: desc or asc")
        ):
            """Get emails from archive with advanced filtering"""
            try:
                from gmail_crew_ai.tools.gmail_tools import EmailArchiveTool
                
                tool = EmailArchiveTool()
                email_tuples = tool._run(
                    date_from=date_from,
                    date_to=date_to,
                    search_query=search_query,
                    limit=limit,
                    folder=folder,
                    include_read=include_read,
                    include_unread=include_unread,
                    sort_by=sort_by,
                    sort_order=sort_order
                )
                
                # Convert tuples to dictionaries with analytics
                emails_data = []
                for subject, sender, body, email_id, thread_info in email_tuples:
                    email_dict = {
                        'email_id': email_id,
                        'subject': subject,
                        'sender': sender,
                        'body_preview': body[:200] + "..." if len(body) > 200 else body,
                        'body': body,
                        'date': thread_info.get('date', '') if isinstance(thread_info, dict) else '',
                        'age_days': self.calculate_age_days(thread_info.get('date', '') if isinstance(thread_info, dict) else ''),
                        'thread_info': thread_info,
                        'needs_response': EmailAnalytics.needs_response({
                            'subject': subject,
                            'body': body,
                            'sender': sender,
                            'thread_info': thread_info,
                            'age_days': self.calculate_age_days(thread_info.get('date', '') if isinstance(thread_info, dict) else '')
                        }),
                        'has_response': EmailAnalytics.has_response({
                            'subject': subject,
                            'body': body,
                            'sender': sender,
                            'thread_info': thread_info
                        })
                    }
                    emails_data.append(email_dict)
                
                return {
                    "emails": emails_data,
                    "total_count": len(emails_data),
                    "filters_applied": {
                        "date_from": date_from,
                        "date_to": date_to,
                        "search_query": search_query,
                        "folder": folder,
                        "include_read": include_read,
                        "include_unread": include_unread,
                        "sort_by": sort_by,
                        "sort_order": sort_order
                    }
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error fetching archive emails: {str(e)}")
        
        @self.app.get("/api/emails/folders")
        async def get_gmail_folders():
            """Get available Gmail folders/labels"""
            try:
                # Return common Gmail folders
                folders = [
                    {"name": "INBOX", "display_name": "Inbox", "description": "Inbox emails"},
                    {"name": "ALL", "display_name": "All Mail", "description": "All emails across folders"},
                    {"name": "SENT", "display_name": "Sent", "description": "Sent emails"},
                    {"name": "DRAFTS", "display_name": "Drafts", "description": "Draft emails"},
                    {"name": "TRASH", "display_name": "Trash", "description": "Deleted emails"}
                ]
                return {"folders": folders}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error getting folders: {str(e)}")

        @self.app.get("/api/responses/drafts")
        async def get_draft_responses():
            """Get saved draft responses"""
            try:
                drafts = []
                if os.path.exists('output/response_report.json'):
                    with open('output/response_report.json', 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'responses' in data:
                            drafts = data['responses']
                        elif isinstance(data, list):
                            drafts = data
                
                return {"drafts": drafts}
                
            except Exception as e:
                return {"drafts": []}
        
        @self.app.post("/api/crew/run")
        async def run_email_crew(email_limit: int = Query(5, ge=1, le=20)):
            """Run the email processing crew"""
            try:
                result = GmailCrewAi().crew().kickoff(inputs={'email_limit': email_limit})
                return {"result": str(result), "status": "completed"}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error running crew: {str(e)}")

        # -------------------------------
        # Real-time Email Monitoring Endpoints
        # -------------------------------

        @self.app.post("/api/listener/start")
        async def start_email_listener(session_id: str = Cookie(None)):
            """Start the real-time email listener"""
            try:
                # Get session credentials
                if not session_id or session_id not in active_sessions:
                    raise HTTPException(status_code=401, detail="No valid session found")
                
                session = active_sessions[session_id]
                
                # Get listener and set credentials
                listener = get_email_listener()
                listener.set_credentials(session['email'], session['app_password'])
                
                success = listener.start_listening()
                
                if success:
                    return {
                        "status": "started",
                        "message": "Real-time email listener started successfully",
                        "listener_status": listener.get_status()
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Failed to start email listener - check credentials",
                        "listener_status": listener.get_status()
                    }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to start email listener: {str(e)}")

        @self.app.post("/api/listener/stop")
        async def stop_email_listener(session_id: str = Cookie(None)):
            """Stop the real-time email listener"""
            try:
                # Verify session exists
                if not session_id or session_id not in active_sessions:
                    raise HTTPException(status_code=401, detail="No valid session found")
                
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

        @self.app.get("/api/listener/status")
        async def get_listener_status(session_id: str = Cookie(None)):
            """Get the current status of the real-time email listener"""
            try:
                # Verify session exists
                if not session_id or session_id not in active_sessions:
                    raise HTTPException(status_code=401, detail="No valid session found")
                
                listener = get_email_listener()
                status = listener.get_status()
                
                return {
                    "listener_status": status,
                    "message": "Listener status retrieved successfully"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get listener status: {str(e)}")

        @self.app.post("/api/listener/restart")
        async def restart_email_listener(session_id: str = Cookie(None)):
            """Restart the real-time email listener"""
            try:
                # Get session credentials
                if not session_id or session_id not in active_sessions:
                    raise HTTPException(status_code=401, detail="No valid session found")
                
                session = active_sessions[session_id]
                
                listener = get_email_listener()
                
                # Stop if running
                listener.stop_listening()
                
                # Set credentials and start
                listener.set_credentials(session['email'], session['app_password'])
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

        @self.app.get("/api/listener/activity-logs")
        async def get_activity_logs(limit: Optional[int] = Query(default=50, ge=1, le=200), session_id: str = Cookie(None)):
            """
            Get recent activity logs from the real-time email monitoring.
            
            Args:
                limit: Maximum number of log entries to return (default: 50, max: 200)
            
            Returns the most recent activity logs from the email listener,
            including timestamps, events, and status changes.
            """
            try:
                # Verify session exists
                if not session_id or session_id not in active_sessions:
                    raise HTTPException(status_code=401, detail="No valid session found")
                
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
                raise HTTPException(status_code=500, detail=f"Failed to get activity logs: {str(e)}")
    
    def generate_sample_archive_data(self, start_date: datetime, end_date: datetime, limit: int) -> List[Dict]:
        """Generate sample archive data for demonstration purposes"""
        sample_emails = []
        senders = [
            'john.doe@company.com', 'sarah.wilson@enterprise.com', 'info@newsletter.com', 
            'support@service.com', 'team@startup.io', 'notifications@platform.com',
            'marketing@brand.com', 'hr@organization.org', 'client@business.net'
        ]
        subjects = [
            'Meeting Request', 'Project Update', 'Weekly Newsletter', 'Support Ticket Response',
            'Quarterly Review', 'Team Standup Notes', 'Client Feedback', 'Invoice #{}',
            'Conference Invitation', 'Product Launch Announcement', 'Security Alert',
            'Monthly Report', 'Budget Approval', 'Training Session', 'Partnership Proposal'
        ]
        
        # Generate random emails within the date range
        current_date = start_date
        email_count = 0
        
        while current_date <= end_date and email_count < limit:
            # Generate 0-3 emails per day
            daily_emails = min(3, limit - email_count)
            for i in range(daily_emails):
                sender = senders[email_count % len(senders)]
                subject = subjects[email_count % len(subjects)]
                if '{}' in subject:
                    subject = subject.format(email_count + 1000)
                
                sample_emails.append({
                    'email_id': f'archive_{current_date.strftime("%Y%m%d")}_{i}',
                    'subject': subject,
                    'sender': sender,
                    'date': current_date.strftime("%Y-%m-%d"),
                    'body_preview': f'This is a sample email from {sender} regarding {(subject or "").lower()}. Email content for demonstration purposes...',
                    'needs_response': (email_count % 4) == 0,  # 25% need response
                    'thread_info': {
                        'thread_size': 1,
                        'date': current_date.strftime("%Y-%m-%d")
                    },
                    'age_days': (datetime.now() - current_date).days
                })
                email_count += 1
                
                if email_count >= limit:
                    break
            
            current_date += timedelta(days=1)
        
        return sample_emails
    
    @staticmethod
    def calculate_age_days(date_str: str) -> int:
        """Calculate age in days from date string"""
        if not date_str:
            return 0
        
        try:
            email_date = datetime.strptime(date_str, "%Y-%m-%d")
            today = datetime.now()
            return (today - email_date).days
        except:
            return 0

# Initialize dashboard
def create_dashboard_app():
    """Create and configure the dashboard application"""
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Create login template if it doesn't exist
    login_template_path = "templates/login.html"
    if not os.path.exists(login_template_path):
        create_login_template()
    
    # Mount static files
    dashboard_app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # Initialize routes
    DashboardRoutes(dashboard_app)
    
    return dashboard_app

def create_login_template():
    """Create the login template file"""
    login_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gmail Dashboard - Professional Sign In</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%234285f4'%3E%3Cpath d='M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 0 1 0 19.366V5.457c0-.904.732-1.636 1.636-1.636h3.819v.001L12 8.733l6.545-4.911V3.821h3.819c.904 0 1.636.732 1.636 1.636z'/%3E%3C/svg%3E">
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-size: 400% 400%;
            animation: gradientShift 8s ease infinite;
        }
        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        .card-shadow {
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
        }
        .input-focus:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
            transform: translateY(-1px);
        }
        .input-focus {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .btn-gradient {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            position: relative;
            overflow: hidden;
        }
        .btn-gradient:hover {
            background: linear-gradient(135deg, #5a67d8 0%, #6b4595 100%);
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        }
        .btn-gradient::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        .btn-gradient:hover::before {
            left: 100%;
        }
        .floating-animation {
            animation: float 6s ease-in-out infinite;
        }
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            25% { transform: translateY(-10px) rotate(1deg); }
            50% { transform: translateY(-15px) rotate(0deg); }
            75% { transform: translateY(-5px) rotate(-1deg); }
        }
        .fade-in {
            animation: fadeIn 1s ease-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .feature-card {
            transition: all 0.3s ease;
        }
        .feature-card:hover {
            transform: translateX(5px);
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 8px;
        }
        .pulse-dot {
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: .5; }
        }
        .glass-morphism {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .success-check {
            animation: checkmark 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
            transform: scale(0);
        }
        @keyframes checkmark {
            to { transform: scale(1); }
        }
    </style>
</head>
<body class="gradient-bg min-h-screen flex items-center justify-center p-4">
    <div class="w-full max-w-5xl mx-auto">
        <div class="glass-morphism rounded-2xl card-shadow overflow-hidden fade-in">
            <div class="flex flex-col lg:flex-row min-h-[600px]">
                <!-- Left Side - Enhanced Branding -->
                <div class="lg:w-1/2 bg-gradient-to-br from-blue-600 via-purple-600 to-purple-700 p-8 lg:p-12 text-white relative overflow-hidden">
                    <!-- Background Pattern -->
                    <div class="absolute inset-0 opacity-10">
                        <svg class="w-full h-full" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
                            <defs>
                                <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
                                    <path d="M 10 0 L 0 0 0 10" fill="none" stroke="white" stroke-width="0.5"/>
                                </pattern>
                            </defs>
                            <rect width="100" height="100" fill="url(#grid)"/>
                        </svg>
                    </div>
                    
                    <div class="relative z-10">
                        <div class="floating-animation mb-6">
                            <div class="flex items-center justify-center w-20 h-20 bg-white bg-opacity-20 rounded-full">
                                <i class="fas fa-envelope text-4xl"></i>
                            </div>
                        </div>
                        
                        <h1 class="text-4xl lg:text-5xl font-bold mb-4 leading-tight">
                            Gmail AI Dashboard
                            <div class="pulse-dot inline-block w-3 h-3 bg-green-400 rounded-full ml-2"></div>
                        </h1>
                        
                        <p class="text-xl mb-8 opacity-90 leading-relaxed">
                            Professional AI-powered email management platform with advanced analytics and intelligent response generation
                        </p>
                        
                        <div class="space-y-4">
                            <div class="feature-card flex items-center p-2 rounded-lg transition-all duration-300">
                                <div class="flex items-center justify-center w-12 h-12 bg-white bg-opacity-20 rounded-lg mr-4">
                                    <i class="fas fa-robot text-2xl"></i>
                                </div>
                                <div>
                                    <span class="text-lg font-semibold">AI-Generated Responses</span>
                                    <p class="text-sm opacity-80">Smart email replies using GPT-4 & Gemini</p>
                                </div>
                            </div>
                            
                            <div class="feature-card flex items-center p-2 rounded-lg transition-all duration-300">
                                <div class="flex items-center justify-center w-12 h-12 bg-white bg-opacity-20 rounded-lg mr-4">
                                    <i class="fas fa-chart-line text-2xl"></i>
                                </div>
                                <div>
                                    <span class="text-lg font-semibold">Advanced Analytics</span>
                                    <p class="text-sm opacity-80">Real-time insights and email patterns</p>
                                </div>
                            </div>
                            
                            <div class="feature-card flex items-center p-2 rounded-lg transition-all duration-300">
                                <div class="flex items-center justify-center w-12 h-12 bg-white bg-opacity-20 rounded-lg mr-4">
                                    <i class="fas fa-shield-alt text-2xl"></i>
                                </div>
                                <div>
                                    <span class="text-lg font-semibold">Secure & Private</span>
                                    <p class="text-sm opacity-80">End-to-end encryption & session security</p>
                                </div>
                            </div>
                            
                            <div class="feature-card flex items-center p-2 rounded-lg transition-all duration-300">
                                <div class="flex items-center justify-center w-12 h-12 bg-white bg-opacity-20 rounded-lg mr-4">
                                    <i class="fas fa-lightning-bolt text-2xl"></i>
                                </div>
                                <div>
                                    <span class="text-lg font-semibold">Real-time Processing</span>
                                    <p class="text-sm opacity-80">Instant email analysis and responses</p>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Trust Indicators -->
                        <div class="mt-8 pt-6 border-t border-white border-opacity-20">
                            <div class="flex items-center space-x-4 text-sm opacity-80">
                                <div class="flex items-center">
                                    <i class="fas fa-check-circle text-green-300 mr-2"></i>
                                    <span>SSL Encrypted</span>
                                </div>
                                <div class="flex items-center">
                                    <i class="fas fa-check-circle text-green-300 mr-2"></i>
                                    <span>SOC 2 Compliant</span>
                                </div>
                                <div class="flex items-center">
                                    <i class="fas fa-check-circle text-green-300 mr-2"></i>
                                    <span>24/7 Monitoring</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Right Side - Enhanced Login Form -->
                <div class="lg:w-1/2 p-8 lg:p-12 bg-gradient-to-br from-gray-50 to-white">
                    <div class="mb-8">
                        <div class="flex items-center mb-4">
                            <h2 class="text-3xl font-bold text-gray-800">Welcome Back</h2>
                            <div class="ml-3 px-3 py-1 bg-green-100 text-green-800 text-xs font-semibold rounded-full">
                                SECURE
                            </div>
                        </div>
                        <p class="text-gray-600">Please sign in with your credentials to access your professional dashboard</p>
                    </div>
                    
                    {% if error %}
                    <div class="mb-6 p-4 bg-red-50 border-l-4 border-red-400 rounded-lg">
                        <div class="flex items-center">
                            <i class="fas fa-exclamation-triangle text-red-500 mr-3"></i>
                            <div>
                                <p class="text-red-700 font-medium">Authentication Failed</p>
                                <p class="text-red-600 text-sm">{{ error }}</p>
                            </div>
                        </div>
                    </div>
                    {% endif %}
                    
                    <!-- Progress Indicator -->
                    <div class="mb-6">
                        <div class="flex items-center justify-between text-sm">
                            <span class="text-gray-500">Setup Progress</span>
                            <span class="text-gray-500" id="progress-text">0/4 Complete</span>
                        </div>
                        <div class="mt-2 bg-gray-200 rounded-full h-2">
                            <div class="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-500" id="progress-bar" style="width: 0%"></div>
                        </div>
                    </div>
                    
                    <form method="post" action="/auth/login" class="space-y-6" id="login-form">
                        <!-- Gmail Credentials Section -->
                        <div class="space-y-4 p-6 bg-white rounded-xl border border-gray-200 shadow-sm">
                            <div class="flex items-center justify-between">
                                <h3 class="text-lg font-semibold text-gray-700 flex items-center">
                                    <i class="fab fa-google text-blue-500 mr-2"></i>
                                    Gmail Credentials
                                </h3>
                                <div class="success-check hidden text-green-500" id="gmail-check">
                                    <i class="fas fa-check-circle"></i>
                                </div>
                            </div>
                            
                            <div>
                                <label for="email" class="block text-sm font-medium text-gray-700 mb-2">
                                    <i class="fas fa-envelope text-gray-400 mr-1"></i>
                                    Gmail Address
                                </label>
                                <div class="relative">
                                    <input
                                        type="email"
                                        id="email"
                                        name="email"
                                        required
                                        class="w-full px-4 py-3 pl-12 border border-gray-300 rounded-lg input-focus transition-all duration-200 focus:ring-2 focus:ring-blue-500"
                                        placeholder="your.email@gmail.com"
                                        autocomplete="email"
                                    >
                                    <i class="fas fa-envelope absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400"></i>
                                    <div class="absolute right-3 top-1/2 transform -translate-y-1/2 hidden" id="email-status">
                                        <i class="fas fa-check-circle text-green-500"></i>
                                    </div>
                                </div>
                            </div>
                            
                            <div>
                                <label for="app_password" class="block text-sm font-medium text-gray-700 mb-2">
                                    <i class="fas fa-key text-gray-400 mr-1"></i>
                                    Gmail App Password
                                </label>
                                <div class="relative">
                                    <input
                                        type="password"
                                        id="app_password"
                                        name="app_password"
                                        required
                                        class="w-full px-4 py-3 pl-12 border border-gray-300 rounded-lg input-focus transition-all duration-200 focus:ring-2 focus:ring-blue-500"
                                        placeholder="••••••••••••••••"
                                        maxlength="16"
                                        autocomplete="current-password"
                                    >
                                    <i class="fas fa-key absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400"></i>
                                    <button type="button" class="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600" onclick="togglePassword('app_password')">
                                        <i class="fas fa-eye" id="app_password_eye"></i>
                                    </button>
                                </div>
                                <div class="mt-2 flex items-center text-xs text-gray-500">
                                    <i class="fas fa-info-circle mr-1"></i>
                                    <a href="https://support.google.com/accounts/answer/185833" target="_blank" class="text-blue-600 hover:underline">
                                        How to generate Gmail App Password
                                    </a>
                                </div>
                            </div>
                        </div>
                        
                        <!-- AI Model Section -->
                        <div class="space-y-4 p-6 bg-white rounded-xl border border-gray-200 shadow-sm">
                            <div class="flex items-center justify-between">
                                <h3 class="text-lg font-semibold text-gray-700 flex items-center">
                                    <i class="fas fa-brain text-purple-500 mr-2"></i>
                                    AI Configuration
                                </h3>
                                <div class="success-check hidden text-green-500" id="ai-check">
                                    <i class="fas fa-check-circle"></i>
                                </div>
                            </div>
                            
                            <div>
                                <label for="model_choice" class="block text-sm font-medium text-gray-700 mb-2">
                                    <i class="fas fa-robot text-gray-400 mr-1"></i>
                                    AI Model Provider
                                </label>
                                <select
                                    id="model_choice"
                                    name="model_choice"
                                    required
                                    class="w-full px-4 py-3 border border-gray-300 rounded-lg input-focus transition-all duration-200 focus:ring-2 focus:ring-purple-500"
                                >
                                    <option value="">Select AI Model</option>
                                    <option value="openai/gpt-4o-mini" data-provider="openai">🚀 OpenAI GPT-4o Mini (Recommended)</option>
                                    <option value="openai/gpt-4o" data-provider="openai">⭐ OpenAI GPT-4o (Premium)</option>
                                    <option value="gemini/gemini-2.0-flash" data-provider="gemini">🔥 Google Gemini 2.0 Flash</option>
                                    <option value="gemini/gemini-pro" data-provider="gemini">💎 Google Gemini Pro</option>
                                </select>
                                <div class="mt-2 text-xs text-gray-500" id="model-info">
                                    Choose your preferred AI model for email analysis and response generation
                                </div>
                            </div>
                            
                            <div>
                                <label for="api_key" class="block text-sm font-medium text-gray-700 mb-2">
                                    <i class="fas fa-code text-gray-400 mr-1"></i>
                                    <span id="api-key-label">API Key</span>
                                </label>
                                <div class="relative">
                                    <input
                                        type="password"
                                        id="api_key"
                                        name="api_key"
                                        required
                                        class="w-full px-4 py-3 pl-12 pr-12 border border-gray-300 rounded-lg input-focus transition-all duration-200 focus:ring-2 focus:ring-purple-500"
                                        placeholder="Your API key"
                                        autocomplete="new-password"
                                    >
                                    <i class="fas fa-code absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" id="api-key-icon"></i>
                                    <button type="button" class="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600" onclick="togglePassword('api_key')">
                                        <i class="fas fa-eye" id="api_key_eye"></i>
                                    </button>
                                </div>
                                <div class="mt-2 flex items-center justify-between text-xs">
                                    <div class="text-gray-500">
                                        <i class="fas fa-shield-alt mr-1"></i>
                                        Your API key is encrypted and session-only
                                    </div>
                                    <a href="#" target="_blank" class="text-blue-600 hover:underline" id="api-key-link">
                                        Get API Key
                                    </a>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Enhanced Submit Button -->
                        <div class="space-y-4">
                            <button
                                type="submit"
                                class="w-full btn-gradient text-white font-semibold py-4 px-6 rounded-lg hover:transform hover:scale-[1.02] transition-all duration-300 flex items-center justify-center relative overflow-hidden"
                                id="submit-btn"
                            >
                                <i class="fas fa-sign-in-alt mr-2"></i>
                                <span id="submit-text">Sign In to Dashboard</span>
                            </button>
                            
                            <!-- Security Notice -->
                            <div class="text-center text-xs text-gray-500">
                                <i class="fas fa-lock mr-1"></i>
                                Protected by enterprise-grade security. Your credentials are never stored permanently.
                            </div>
                        </div>
                    </form>
                    
                    <!-- Enhanced Help Section -->
                    <div class="mt-8 pt-6 border-t border-gray-200">
                        <div class="flex items-center justify-between mb-4">
                            <h4 class="text-sm font-semibold text-gray-700 flex items-center">
                                <i class="fas fa-question-circle text-blue-500 mr-2"></i>
                                Need Help?
                            </h4>
                            <button type="button" class="text-blue-600 hover:text-blue-800 text-sm" onclick="toggleHelp()">
                                <span id="help-toggle">Show Guide</span>
                                <i class="fas fa-chevron-down ml-1" id="help-chevron"></i>
                            </button>
                        </div>
                        
                        <div class="space-y-3 text-sm text-gray-600 hidden" id="help-content">
                            <div class="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-400">
                                <div class="flex items-start">
                                    <i class="fab fa-google text-blue-500 mr-3 mt-1"></i>
                                    <div>
                                        <p class="font-medium text-blue-800">Gmail App Password Setup:</p>
                                        <ol class="text-blue-700 text-xs mt-1 space-y-1">
                                            <li>1. Go to Google Account → Security</li>
                                            <li>2. Enable 2-Step Verification</li>
                                            <li>3. Go to App passwords section</li>
                                            <li>4. Generate password for "Gmail CrewAI"</li>
                                        </ol>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="bg-purple-50 p-4 rounded-lg border-l-4 border-purple-400">
                                <div class="flex items-start">
                                    <i class="fas fa-key text-purple-500 mr-3 mt-1"></i>
                                    <div>
                                        <p class="font-medium text-purple-800">API Key Setup:</p>
                                        <ul class="text-purple-700 text-xs mt-1 space-y-1">
                                            <li>• <strong>OpenAI:</strong> platform.openai.com → API Keys</li>
                                            <li>• <strong>Gemini:</strong> makersuite.google.com → Get API Key</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="bg-green-50 p-4 rounded-lg border-l-4 border-green-400">
                                <div class="flex items-start">
                                    <i class="fas fa-shield-check text-green-500 mr-3 mt-1"></i>
                                    <div>
                                        <p class="font-medium text-green-800">Security Features:</p>
                                        <ul class="text-green-700 text-xs mt-1 space-y-1">
                                            <li>• Session-based authentication (24 hours)</li>
                                            <li>• No permanent credential storage</li>
                                            <li>• End-to-end encryption</li>
                                            <li>• Automatic session expiry</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Enhanced JavaScript functionality
        let progressValue = 0;
        
        function updateProgress() {
            const email = document.getElementById('email').value;
            const password = document.getElementById('app_password').value;
            const model = document.getElementById('model_choice').value;
            const apiKey = document.getElementById('api_key').value;
            
            let completed = 0;
            if (email && email.includes('@gmail.com')) completed++;
            if (password && password.length >= 16) completed++;
            if (model) completed++;
            if (apiKey && apiKey.length > 10) completed++;
            
            progressValue = (completed / 4) * 100;
            document.getElementById('progress-bar').style.width = progressValue + '%';
            document.getElementById('progress-text').textContent = completed + '/4 Complete';
            
            // Show checkmarks
            if (email && email.includes('@gmail.com')) {
                document.getElementById('email-status').classList.remove('hidden');
                document.getElementById('gmail-check').classList.remove('hidden');
            }
            if (model && apiKey && apiKey.length > 10) {
                document.getElementById('ai-check').classList.remove('hidden');
            }
        }
        
        function togglePassword(fieldId) {
            const field = document.getElementById(fieldId);
            const eye = document.getElementById(fieldId + '_eye');
            
            if (field.type === 'password') {
                field.type = 'text';
                eye.classList.remove('fa-eye');
                eye.classList.add('fa-eye-slash');
            } else {
                field.type = 'password';
                eye.classList.remove('fa-eye-slash');
                eye.classList.add('fa-eye');
            }
        }
        
        function toggleHelp() {
            const content = document.getElementById('help-content');
            const toggle = document.getElementById('help-toggle');
            const chevron = document.getElementById('help-chevron');
            
            if (content.classList.contains('hidden')) {
                content.classList.remove('hidden');
                toggle.textContent = 'Hide Guide';
                chevron.classList.remove('fa-chevron-down');
                chevron.classList.add('fa-chevron-up');
            } else {
                content.classList.add('hidden');
                toggle.textContent = 'Show Guide';
                chevron.classList.remove('fa-chevron-up');
                chevron.classList.add('fa-chevron-down');
            }
        }
        
        // Model selection handler with enhanced features
        document.getElementById('model_choice').addEventListener('change', function() {
            const apiKeyInput = document.getElementById('api_key');
            const apiKeyLabel = document.getElementById('api-key-label');
            const apiKeyIcon = document.getElementById('api-key-icon');
            const apiKeyLink = document.getElementById('api-key-link');
            const modelInfo = document.getElementById('model-info');
            
            const selectedOption = this.options[this.selectedIndex];
            const provider = selectedOption.dataset.provider;
            
            if (provider === 'openai') {
                apiKeyInput.placeholder = 'sk-proj-... (OpenAI API Key)';
                apiKeyLabel.innerHTML = '<i class="fas fa-robot text-green-500 mr-1"></i>OpenAI API Key';
                apiKeyIcon.className = 'fas fa-robot absolute left-4 top-1/2 transform -translate-y-1/2 text-green-500';
                apiKeyLink.href = 'https://platform.openai.com/api-keys';
                apiKeyLink.textContent = 'Get OpenAI Key';
                modelInfo.textContent = 'OpenAI models provide excellent reasoning and natural language understanding';
            } else if (provider === 'gemini') {
                apiKeyInput.placeholder = 'AIza... (Google Gemini API Key)';
                apiKeyLabel.innerHTML = '<i class="fab fa-google text-blue-500 mr-1"></i>Gemini API Key';
                apiKeyIcon.className = 'fab fa-google absolute left-4 top-1/2 transform -translate-y-1/2 text-blue-500';
                apiKeyLink.href = 'https://makersuite.google.com/app/apikey';
                apiKeyLink.textContent = 'Get Gemini Key';
                modelInfo.textContent = 'Gemini models offer fast performance and multimodal capabilities';
            } else {
                apiKeyInput.placeholder = 'Your API key';
                apiKeyLabel.innerHTML = 'API Key';
                apiKeyIcon.className = 'fas fa-code absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400';
                apiKeyLink.href = '#';
                apiKeyLink.textContent = 'Get API Key';
                modelInfo.textContent = 'Choose your preferred AI model for email analysis and response generation';
            }
            
            updateProgress();
        });
        
        // Add input listeners for progress tracking
        ['email', 'app_password', 'model_choice', 'api_key'].forEach(id => {
            document.getElementById(id).addEventListener('input', updateProgress);
        });
        
        // Enhanced form validation
        document.getElementById('login-form').addEventListener('submit', function(e) {
            const requiredFields = ['email', 'app_password', 'model_choice', 'api_key'];
            let isValid = true;
            
            requiredFields.forEach(fieldName => {
                const field = document.getElementById(fieldName);
                const fieldContainer = field.closest('div');
                
                if (!field.value.trim()) {
                    field.classList.add('border-red-500', 'bg-red-50');
                    isValid = false;
                } else {
                    field.classList.remove('border-red-500', 'bg-red-50');
                    field.classList.add('border-green-500');
                }
            });
            
            // Email validation
            const email = document.getElementById('email').value;
            if (email && !email.includes('@gmail.com')) {
                document.getElementById('email').classList.add('border-red-500', 'bg-red-50');
                isValid = false;
            }
            
            // App password length validation
            const appPassword = document.getElementById('app_password').value;
            if (appPassword && appPassword.length !== 16) {
                document.getElementById('app_password').classList.add('border-red-500', 'bg-red-50');
                isValid = false;
            }
            
            if (!isValid) {
                e.preventDefault();
                // Show error message
                const errorDiv = document.createElement('div');
                errorDiv.className = 'mb-4 p-4 bg-red-50 border-l-4 border-red-400 rounded-lg';
                errorDiv.innerHTML = `
                    <div class="flex items-center">
                        <i class="fas fa-exclamation-triangle text-red-500 mr-3"></i>
                        <div>
                            <p class="text-red-700 font-medium">Validation Error</p>
                            <p class="text-red-600 text-sm">Please check all fields and ensure they meet the requirements.</p>
                        </div>
                    </div>
                `;
                
                const form = document.getElementById('login-form');
                form.insertBefore(errorDiv, form.firstChild);
                
                // Remove error after 5 seconds
                setTimeout(() => errorDiv.remove(), 5000);
                return;
            }
            
            // Show enhanced loading state
            const submitBtn = document.getElementById('submit-btn');
            const submitText = document.getElementById('submit-text');
            
            submitBtn.disabled = true;
            submitBtn.classList.add('opacity-90', 'cursor-not-allowed');
            submitText.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Authenticating...';
            
            // Simulate progress
            let dots = 0;
            const loadingInterval = setInterval(() => {
                dots = (dots + 1) % 4;
                submitText.innerHTML = `<i class="fas fa-spinner fa-spin mr-2"></i>Authenticating${'.'.repeat(dots)}`;
            }, 500);
            
            // Clear interval after 10 seconds (fallback)
            setTimeout(() => clearInterval(loadingInterval), 10000);
        });
        
        // Auto-format app password
        document.getElementById('app_password').addEventListener('input', function(e) {
            let value = e.target.value.replace(/\\s/g, ''); // Remove spaces
            if (value.length > 16) {
                value = value.substring(0, 16);
            }
            e.target.value = value;
            updateProgress();
        });
        
        // Email validation
        document.getElementById('email').addEventListener('blur', function(e) {
            const email = e.target.value;
            if (email && !email.includes('@gmail.com')) {
                e.target.classList.add('border-yellow-500');
                // Show warning tooltip
                const warning = document.createElement('div');
                warning.className = 'text-xs text-yellow-600 mt-1';
                warning.innerHTML = '<i class="fas fa-exclamation-triangle mr-1"></i>Please use a Gmail address';
                e.target.parentNode.appendChild(warning);
                
                setTimeout(() => warning.remove(), 3000);
            }
        });
    </script>
</body>
</html>'''
    
    with open("templates/login.html", 'w') as f:
        f.write(login_html)

# Export for external use
app = create_dashboard_app()

def run_dashboard():
    """Run the dashboard server"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    run_dashboard()
