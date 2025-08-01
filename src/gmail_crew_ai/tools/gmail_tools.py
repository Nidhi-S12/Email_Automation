# ============================================================
# Gmail Tools: Fetching, Drafting, Organizing, and Deleting
# ============================================================

import imaplib
import email
import email.utils
from email.header import decode_header
from typing import List, Tuple, Literal, Optional, Type, Dict, Any, Callable
import re
from bs4 import BeautifulSoup
from crewai.tools import BaseTool
import os
from pydantic import BaseModel, Field
from crewai.tools import tool
import time
import datetime  # Add this import
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import base64
import threading
import asyncio
import select
import socket
import json
import traceback
from datetime import datetime as dt

# -------------------------------
# Utility Functions
# -------------------------------

def decode_header_safe(header):
    """
    Safely decode email headers that might contain encoded words or non-ASCII characters.
    """
    if header is None:
        return ""
    try:
        decoded_parts = []
        for decoded_str, charset in decode_header(header):
            if isinstance(decoded_str, bytes):
                if charset:
                    decoded_parts.append(decoded_str.decode(charset or 'utf-8', errors='replace'))
                else:
                    decoded_parts.append(decoded_str.decode('utf-8', errors='replace'))
            else:
                decoded_parts.append(str(decoded_str))
        return ' '.join(decoded_parts)
    except Exception as e:
        # Fallback to raw header if decoding fails
        return str(header) if header is not None else ""

def clean_email_body(email_body: str) -> str:
    """
    Clean the email body by removing HTML tags and excessive whitespace.
    """
    try:
        soup = BeautifulSoup(email_body, "html.parser")
        text = soup.get_text(separator=" ")  # Get text with spaces instead of <br/>
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        text = email_body  # Fallback to raw body if parsing fails
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_thread_history(thread_history: Dict[str, Any]) -> str:
    """
    Format thread history into a readable string.
    """
    if not thread_history or "error" in thread_history:
        return f"Error retrieving thread history: {thread_history.get('error', 'Unknown error')}"
    
    messages = thread_history.get("thread_messages", [])
    if not messages:
        return "No messages found in thread history."
    
    formatted = f"Thread: {thread_history.get('thread_subject', 'No Subject')}\n"
    formatted += f"Participants: {', '.join(thread_history.get('participants', []))}\n"
    formatted += f"Total Messages: {thread_history.get('message_count', 0)}\n\n"
    
    # Add each message in chronological order
    for i, msg in enumerate(messages, 1):
        formatted += f"--- Message {i} ---\n"
        formatted += f"From: {msg.get('sender', 'Unknown')}\n"
        formatted += f"To: {msg.get('to', 'Unknown')}\n"
        formatted += f"Date: {msg.get('date', 'Unknown')}\n"
        formatted += f"Subject: {msg.get('subject', 'No Subject')}\n"
        
        # Add attachment info if available
        if 'attachments' in msg and msg['attachments']:
            formatted += "Attachments:\n"
            for att in msg['attachments']:
                formatted += f"  - {att.get('filename', 'Unknown')} ({att.get('type', 'Unknown')}, {att.get('size', 0)} bytes)\n"
        formatted += f"\n{msg.get('body', 'No content')}\n\n"
    
    return formatted

# -------------------------------
# Base Gmail Tool
# -------------------------------

class GmailToolBase(BaseTool):
    """Base class for Gmail tools, handling connection and credentials."""
    class Config:
        arbitrary_types_allowed = True

    email_address: Optional[str] = Field(None, description="Gmail email address")
    app_password: Optional[str] = Field(None, description="Gmail app password")

    def __init__(self, description: str = ""):
        super().__init__(description=description)
        self.email_address = os.environ.get("EMAIL_ADDRESS")
        self.app_password = os.environ.get("APP_PASSWORD")
        if not self.email_address or not self.app_password:
            raise ValueError("EMAIL_ADDRESS and APP_PASSWORD must be set in the environment.")

    def _connect(self):
        """Connect to Gmail."""
        try:
            print(f"Connecting to Gmail with email: {self.email_address[:3]}...{self.email_address[-8:]}")
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(self.email_address, self.app_password)
            print("Successfully logged in to Gmail")
            return mail
        except Exception as e:
            print(f"Error connecting to Gmail: {e}")
            raise e

    def _disconnect(self, mail):
        """Disconnect from Gmail."""
        try:
            mail.close()
            mail.logout()
            print("Successfully disconnected from Gmail")
        except:
            pass

    def _get_thread_messages(self, mail: imaplib.IMAP4_SSL, msg) -> List[Dict[str, Any]]:
        """Get all messages in the thread by following References and In-Reply-To headers."""
        thread_messages = []

        # Get message IDs from References and In-Reply-To headers
        references = msg.get("References", "").split()
        in_reply_to = msg.get("In-Reply-To", "").split()
        current_msg_id = msg.get("Message-ID", "").strip()
        
        # Combine all message IDs, including the current one
        message_ids = list(set(references + in_reply_to + [current_msg_id]))
        message_ids = [mid for mid in message_ids if mid]  # Remove empty strings
        
        if message_ids:
            # Build a complex search query that looks for these message IDs in any relevant header
            search_terms = []
            for mid in message_ids:
                # Escape quotes in message IDs and use simpler search format
                escaped_mid = mid.replace('"', '\\"')
                search_terms.append(f'HEADER "Message-ID" "{escaped_mid}"')
                search_terms.append(f'HEADER "References" "{escaped_mid}"')
                search_terms.append(f'HEADER "In-Reply-To" "{escaped_mid}"')
            
            # Use OR search but limit the complexity to avoid parse errors
            if len(search_terms) <= 10:  # Limit complexity
                search_query = f"({' OR '.join(search_terms)})"
            else:
                # Fallback to simpler search for the primary message ID only
                escaped_primary = message_ids[0].replace('"', '\\"')
                search_query = f'HEADER "Message-ID" "{escaped_primary}"'

            try:
                result, data = mail.search(None, search_query)
                if result == "OK" and data[0]:
                    thread_ids = data[0].split()
                    print(f"Found {len(thread_ids)} messages in thread")
                    
                    for thread_id in thread_ids:
                        result, msg_data = mail.fetch(thread_id, "(RFC822)")
                        if result == "OK":
                            thread_msg = email.message_from_bytes(msg_data[0][1])
                            
                            # Extract useful metadata and body
                            message_info = {
                                "subject": decode_header_safe(thread_msg["Subject"]),
                                "sender": decode_header_safe(thread_msg["From"]),
                                "date": thread_msg["Date"],
                                "body": self._extract_body(thread_msg),
                                "message_id": thread_msg.get("Message-ID", ""),
                                "email_id": thread_id.decode('utf-8')
                            }
                            
                            thread_messages.append(message_info)
            except Exception as e:
                print(f"Error searching for thread messages: {e}")
        
        # Sort messages by date
        thread_messages.sort(key=lambda x: email.utils.parsedate_to_datetime(x['date']) 
                          if x.get('date') and email.utils.parsedate_to_datetime(x['date']) 
                          else datetime.datetime.min)
        return thread_messages

    def _extract_body(self, msg) -> str:
        """Extract body from an email message."""
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                try:
                    email_body = part.get_payload(decode=True).decode()
                except:
                    email_body = ""

                if content_type == "text/plain" and "attachment" not in content_disposition:
                    body += email_body
                elif content_type == "text/html" and "attachment" not in content_disposition:
                    body += clean_email_body(email_body)
        else:
            try:
                body = clean_email_body(msg.get_payload(decode=True).decode())
            except Exception as e:
                body = f"Error decoding body: {e}"
        return body

# -------------------------------
# Get Unread Emails Tool
# -------------------------------

class GetUnreadEmailsSchema(BaseModel):
    """Schema for GetUnreadEmailsTool input."""
    limit: Optional[int] = Field(
        default=5,
        description="Maximum number of unread emails to retrieve. Defaults to 5.",
        ge=1  # Ensures the limit is greater than or equal to 1
    )
    prioritize_primary: Optional[bool] = Field(
        default=True,
        description="Whether to prioritize Primary tab emails over Promotions/Social/Updates. Defaults to True."
    )

class GetUnreadEmailsTool(GmailToolBase):
    """Tool to get unread emails from Gmail."""
    name: str = "get_unread_emails"
    description: str = "Gets unread emails from Gmail"
    args_schema: Type[BaseModel] = GetUnreadEmailsSchema

    def _run(self, limit: Optional[int] = 5, prioritize_primary: Optional[bool] = True) -> List[Tuple[str, str, str, str, Dict]]:
        """
        Fetch unread emails from Gmail up to the specified limit.
        Prioritizes Primary tab emails first, then falls back to other categories if needed.
        Returns a list of tuples: (subject, sender, body, email_id, thread_info)
        """
        mail = self._connect()
        try:
            print("DEBUG: Connecting to Gmail...")
            mail.select("INBOX")
            
            emails = []
            
            if prioritize_primary:
                # First, try to get emails from Primary category
                primary_emails = self._get_emails_by_category(mail, 'PRIMARY', limit)
                emails.extend(primary_emails)
                
                # If we still need more emails and haven't reached the limit
                remaining_limit = limit - len(emails)
                if remaining_limit > 0:
                    print(f"DEBUG: Got {len(primary_emails)} primary emails, fetching {remaining_limit} more from other categories...")
                    # Get remaining emails from other categories
                    other_emails = self._get_emails_by_category(mail, 'OTHER', remaining_limit)
                    emails.extend(other_emails)
            else:
                # Get all emails without prioritization
                emails = self._get_emails_by_category(mail, 'ALL', limit)
            
            print(f"DEBUG: Returning {len(emails)} email tuples")
            return emails[:limit]  # Ensure we don't exceed the limit
            
        except Exception as e:
            print(f"DEBUG: Exception in GetUnreadEmailsTool: {e}")
            traceback.print_exc()
            return []
        finally:
            self._disconnect(mail)

    def _get_emails_by_category(self, mail, category: str, limit: int) -> List[Tuple[str, str, str, str, Dict]]:
        """
        Get emails by category (PRIMARY, OTHER, or ALL)
        """
        emails = []
        
        try:
            if category == 'PRIMARY':
                # Search for unread emails in Primary category
                # Gmail uses X-GM-LABELS for categories, but IMAP search is limited
                # We'll fetch all unread and filter based on common primary indicators
                result, data = mail.search(None, 'UNSEEN')
            elif category == 'OTHER':
                # For non-primary emails, we'll use a different approach
                result, data = mail.search(None, 'UNSEEN')
            else:  # ALL
                result, data = mail.search(None, 'UNSEEN')
            
            if result != "OK":
                print(f"DEBUG: Error searching for {category} emails")
                return []
                
            email_ids = data[0].split()
            print(f"DEBUG: Found {len(email_ids)} unread emails for category {category}")
            
            if not email_ids:
                print(f"DEBUG: No unread emails found in {category}.")
                return []
            
            # Reverse to get newest first
            email_ids = list(reversed(email_ids))
            
            processed_count = 0
            primary_count = 0
            other_count = 0
            
            for email_id in email_ids:
                if processed_count >= limit:
                    break
                    
                try:
                    result, msg_data = mail.fetch(email_id, "(RFC822)")
                    if result != "OK":
                        print(f"Error fetching email {email_id}:", result)
                        continue
                        
                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email)
                    
                    # Determine if this email is primary or not
                    is_primary = self._is_primary_email(msg)
                    
                    # Filter based on category request
                    if category == 'PRIMARY' and not is_primary:
                        other_count += 1
                        continue
                    elif category == 'OTHER' and is_primary:
                        primary_count += 1
                        continue
                    
                    # Process the email
                    email_tuple = self._process_email(mail, email_id, msg)
                    if email_tuple:
                        emails.append(email_tuple)
                        processed_count += 1
                        if is_primary:
                            primary_count += 1
                        else:
                            other_count += 1
                            
                except Exception as e:
                    print(f"Error processing email {email_id}: {e}")
                    continue
            
            print(f"DEBUG: Processed {processed_count} emails for {category} (Primary: {primary_count}, Other: {other_count})")
            return emails
            
        except Exception as e:
            print(f"DEBUG: Exception in _get_emails_by_category for {category}: {e}")
            return []

    def _is_primary_email(self, msg) -> bool:
        """
        Determine if an email belongs to Primary category based on sender patterns and content.
        """
        sender = decode_header_safe(msg.get("From", "")).lower()
        subject = decode_header_safe(msg.get("Subject", "")).lower()
        
        # Common promotional/social/update indicators
        promotional_indicators = [
            'noreply', 'no-reply', 'newsletter', 'unsubscribe', 'marketing',
            'promotion', 'deal', 'offer', 'sale', 'discount', 'coupon',
            'pinterest', 'facebook', 'twitter', 'instagram', 'linkedin',
            'notifications', 'automated', 'bot', 'system',
            'updates@', 'news@', 'alerts@', 'digest'
        ]
        
        # Check if sender contains promotional indicators
        for indicator in promotional_indicators:
            if indicator in sender:
                return False
        
        # Check subject for promotional content
        promotional_subjects = [
            'newsletter', 'weekly', 'daily', 'digest', 'update',
            'sale', 'deal', 'offer', 'promotion', 'discount',
            'unsubscribe', 'notification'
        ]
        
        for indicator in promotional_subjects:
            if indicator in subject:
                return False
        
        # If no promotional indicators found, consider it primary
        return True

    def _process_email(self, mail, email_id, msg) -> Optional[Tuple[str, str, str, str, Dict]]:
        """
        Process a single email and return the email tuple.
        """
        try:
            # Decode headers properly (handles encoded characters)
            subject = decode_header_safe(msg["Subject"])
            sender = decode_header_safe(msg["From"])
            
            # Extract and standardize the date
            date_str = msg.get("Date", "")
            received_date = self._parse_email_date(date_str)
            
            # Get the current message body
            current_body = self._extract_body(msg)
            
            # Get thread messages
            thread_messages = self._get_thread_messages(mail, msg)
            
            # Combine current message with thread history
            full_body = "\n\n--- Previous Messages ---\n".join([current_body] + thread_messages)
            
            # Get thread metadata
            thread_info = {
                'message_id': msg.get('Message-ID', ''),
                'in_reply_to': msg.get('In-Reply-To', ''),
                'references': msg.get('References', ''),
                'date': received_date,  # Use standardized date
                'raw_date': date_str,   # Keep original date string
                'email_id': email_id.decode('utf-8') if isinstance(email_id, bytes) else str(email_id)
            }
            
            # Add a clear date indicator in the body for easier extraction
            full_body = f"EMAIL DATE: {received_date}\n\n{full_body}"
            
            email_id_str = email_id.decode('utf-8') if isinstance(email_id, bytes) else str(email_id)
            return (subject, sender, full_body, email_id_str, thread_info)
            
        except Exception as e:
            print(f"Error processing email: {e}")
            return None

    def _parse_email_date(self, date_str: str) -> str:
        """
        Parse email date string into a standardized format (YYYY-MM-DD).
        """
        if not date_str:
            return ""
        
        try:
            # Try various date formats commonly found in emails
            # Remove timezone name if present (like 'EDT', 'PST')
            date_str = re.sub(r'\s+\([A-Z]{3,4}\)', '', date_str)
            # Parse with email.utils
            parsed_date = email.utils.parsedate_to_datetime(date_str)
            if parsed_date:
                return parsed_date.strftime("%Y-%m-%d")
        except Exception as e:
            print(f"Error parsing date '{date_str}': {e}")
        
        return ""

# -------------------------------
# Save Draft Tool
# -------------------------------

class SaveDraftSchema(BaseModel):
    """Schema for SaveDraftTool input."""
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body content")
    recipient: str = Field(..., description="Recipient email address")
    thread_info: Optional[Dict[str, Any]] = Field(None, description="Thread information for replies")

class SaveDraftTool(BaseTool):
    """Tool to save an email as a draft using IMAP."""
    name: str = "save_email_draft"
    description: str = "Saves an email as a draft in Gmail"
    args_schema: Type[BaseModel] = SaveDraftSchema

    def _format_body(self, body: str) -> str:
        """Format the email body - signature should already be included by the agent."""
        # The agent should handle signature creation directly
        # No hardcoded name replacements - the agent must use the actual user's name
        return body

    def _connect(self):
        """Connect to Gmail using IMAP."""
        # Get email credentials from environment
        email_address = os.environ.get('EMAIL_ADDRESS')
        app_password = os.environ.get('APP_PASSWORD')
        
        if not email_address or not app_password:
            raise ValueError("EMAIL_ADDRESS or APP_PASSWORD environment variables not set")
        
        # Connect to Gmail's IMAP server
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        print(f"Connecting to Gmail with email: {email_address[:3]}...{email_address[-10:]}")
        mail.login(email_address, app_password)
        return mail, email_address

    def _disconnect(self, mail):
        """Disconnect from Gmail."""
        try:
            mail.logout()
        except:
            pass

    def _check_drafts_folder(self, mail):
        """Check available mailboxes to find the drafts folder."""
        print("Checking available mailboxes...")
        result, mailboxes = mail.list()
        if result == 'OK':
            drafts_folders = []
            for mailbox in mailboxes:
                if b'Drafts' in mailbox or b'Draft' in mailbox:
                    drafts_folders.append(mailbox.decode())
                    print(f"Found drafts folder: {mailbox.decode()}")
            return drafts_folders
        return []

    def _verify_draft_saved(self, mail, subject, recipient):
        """Verify if the draft was actually saved by searching for it."""
        try:
            # Try different drafts folder names
            drafts_folders = [
                '"[Gmail]/Drafts"', 
                'Drafts',
                'DRAFTS',
                '"[Google Mail]/Drafts"',
                '[Gmail]/Drafts'
            ]
            for folder in drafts_folders:
                try:
                    print(f"Checking folder: {folder}")
                    result, _ = mail.select(folder, readonly=True)
                    if result != 'OK':
                        continue
                        
                    # Search for drafts with this subject
                    search_criteria = f'SUBJECT "{subject}"'
                    result, data = mail.search(None, search_criteria)
                    
                    if result == 'OK' and data[0]:
                        draft_count = len(data[0].split())
                        print(f"Found {draft_count} drafts matching subject '{subject}' in folder {folder}")
                        return True, folder
                    else:
                        print(f"No drafts found matching subject '{subject}' in folder {folder}")
                except Exception as e:
                    print(f"Error checking folder {folder}: {e}")
                    continue
            return False, None
        except Exception as e:
            print(f"Error verifying draft: {e}")
            return False, None

    def _get_latest_thread_info(self, thread_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure we have the latest message's thread info for proper threading.
        If thread_info has a message_id but no thread_history, fetch the thread.
        """
        if not thread_info:
            return {}
            
        # If we already have full thread history, use the latest message
        if thread_info.get('thread_messages') and len(thread_info['thread_messages']) > 0:
            # Sort messages by date and get the latest one
            sorted_msgs = sorted(
                thread_info['thread_messages'],
                key=lambda x: email.utils.parsedate_to_datetime(x['date']) if x.get('date') else datetime.datetime.min,
                reverse=True
            )
            latest = sorted_msgs[0]
            return {
                'message_id': latest.get('message_id', ''),
                'references': thread_info.get('references', ''),
                'in_reply_to': latest.get('message_id', ''),
                'date': latest.get('date', ''),
                'email_id': thread_info.get('email_id', '')
            }
        
        # If we only have a single message_id, use that
        return thread_info

    def _run(self, subject: str, body: str, recipient: str, thread_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a draft email to Gmail Drafts folder.
        """
        try:
            mail, email_address = self._connect()
            
            # Check available drafts folders
            drafts_folders = self._check_drafts_folder(mail)
            print(f"Available drafts folders: {drafts_folders}")
            
            # Try with quoted folder name first
            drafts_folder = '"[Gmail]/Drafts"'
            print(f"Selecting drafts folder: {drafts_folder}")
            result, _ = mail.select(drafts_folder)
            
            # If that fails, try without quotes
            if result != 'OK':
                drafts_folder = '[Gmail]/Drafts'
                print(f"First attempt failed. Trying: {drafts_folder}")
                result, _ = mail.select(drafts_folder)
            
            # If that also fails, try just 'Drafts'
            if result != 'OK':
                drafts_folder = 'Drafts'
                print(f"Second attempt failed. Trying: {drafts_folder}")
                result, _ = mail.select(drafts_folder)
            
            if result != 'OK':
                return f"Error: Could not select drafts folder. Available folders: {drafts_folders}"
                
            print(f"Successfully selected drafts folder: {drafts_folder}")
            
            # Format body and add signature
            body_with_signature = self._format_body(body)
            message = email.message.EmailMessage()
            message["From"] = email_address
            message["To"] = recipient

            # --- Robust subject handling for threading and Gmail compatibility ---
            thread_subject = None
            if thread_info:
                thread_subject = thread_info.get('thread_subject') or None
                # Try to extract from thread_messages if not present
                if not thread_subject and thread_info.get('thread_messages'):
                    for msg in thread_info['thread_messages']:
                        if msg.get('subject'):
                            thread_subject = msg['subject']
                            break

            # Gmail threading: If original subject is missing, use "Re: " only if replying, else "No Subject"
            if not subject or subject.strip() == "":
                if thread_subject and thread_subject.strip():
                    subject = thread_subject.strip()
                elif thread_info:
                    subject = "Re: No Subject"
                else:
                    subject = "No Subject"

            # Always ensure subject is prefixed with "Re: " for replies
            if thread_info and not subject.lower().startswith('re:'):
                subject = f"Re: {subject}"

            message["Subject"] = subject

            # --- Always set threading headers if replying ---
            if thread_info:
                latest_thread_info = self._get_latest_thread_info(thread_info)
                references = []
                if latest_thread_info.get('references'):
                    references.extend(latest_thread_info['references'].split())
                if latest_thread_info.get('message_id'):
                    if latest_thread_info['message_id'] not in references:
                        references.append(latest_thread_info['message_id'])
                if references:
                    message["References"] = " ".join(references)
                if latest_thread_info.get('message_id'):
                    message["In-Reply-To"] = latest_thread_info['message_id']

            message.set_content(body_with_signature)

            # Save to drafts
            print(f"Attempting to save draft to {drafts_folder}...")
            date = imaplib.Time2Internaldate(time.time())
            result, data = mail.append(drafts_folder, '\\Draft', date, message.as_bytes())
            
            if result != 'OK':
                return f"Error saving draft: {result}, {data}"
            
            print(f"Draft save attempt result: {result}")
            
            # Verify the draft was actually saved
            verified, folder = self._verify_draft_saved(mail, subject, recipient)
            
            if verified:
                return f"VERIFIED: Draft email saved with subject: '{subject}' in folder {folder}"
            else:
                # Try Gmail's API approach as a fallback
                try:
                    # Try saving directly to All Mail and flagging as draft
                    result, data = mail.append('[Gmail]/All Mail', '\\Draft', date, message.as_bytes())
                    if result == 'OK':
                        return f"Draft saved to All Mail with subject: '{subject}' (flagged as draft)"
                    else:
                        return f"WARNING: Draft save attempt returned {result}, but verification failed. Please check your Gmail Drafts folder."
                except Exception as e:
                    return f"WARNING: Draft may not have been saved properly: {str(e)}"
        except Exception as e:
            return f"Error saving draft: {str(e)}"
        finally:
            self._disconnect(mail)

# -------------------------------
# Organize Email Tool
# -------------------------------

class GmailOrganizeSchema(BaseModel):
    """Schema for GmailOrganizeTool input."""
    email_id: str = Field(..., description="Email ID to organize")
    category: str = Field(..., description="Category assigned by agent (Urgent/Response Needed/etc)")
    priority: str = Field(..., description="Priority level (High/Medium/Low)")
    should_star: bool = Field(default=False, description="Whether to star the email")
    labels: List[str] = Field(default_list=[], description="Labels to apply")

class GmailOrganizeTool(GmailToolBase):
    """Tool to organize emails based on agent categorization."""
    name: str = "organize_email"
    description: str = "Organizes emails using Gmail's priority features based on category and priority"
    args_schema: Type[BaseModel] = GmailOrganizeSchema

    def _run(self, email_id: str, category: str, priority: str, should_star: bool = False, labels: List[str] = None) -> str:
        """
        Organize an email with the specified parameters.
        """
        if labels is None:
            # Provide a default empty list to avoid validation errors
            labels = []
        print(f"Organizing email {email_id} with category {category}, priority {priority}, star={should_star}, labels={labels}")
        
        mail = self._connect()
        try:
            # Select inbox to ensure we can access the email
            mail.select("INBOX")
            
            # Apply organization based on category and priority
            if category == "Urgent Response Needed" and priority == "High":
                # Star the email
                if should_star:
                    mail.store(email_id, '+FLAGS', '\\Flagged')
                
                # Mark as important
                mail.store(email_id, '+FLAGS', '\\Important')
                
                # Apply URGENT label if it doesn't exist
                if "URGENT" not in labels:
                    labels.append("URGENT")

            # Apply all specified labels
            for label in labels:
                try:
                    # Create label if it doesn't exist
                    mail.create(label)
                except:
                    pass  # Label might already exist
                
                # Apply label
                mail.store(email_id, '+X-GM-LABELS', label)

            return f"Email organized: Starred={should_star}, Labels={labels}"

        except Exception as e:
            return f"Error organizing email: {e}"
        finally:
            self._disconnect(mail)

# -------------------------------
# Delete Email Tool
# -------------------------------

class GmailDeleteSchema(BaseModel):
    """Schema for GmailDeleteTool input."""
    email_id: str = Field(..., description="Email ID to delete")
    reason: str = Field(..., description="Reason for deletion")

class GmailDeleteTool(BaseTool):
    """Tool to delete an email using IMAP."""
    name: str = "delete_email"
    description: str = "Deletes an email from Gmail"

    def _run(self, email_id: str, reason: str) -> str:
        """
        Delete an email by ID.
        Parameters:
            email_id: The email ID to delete
            reason: The reason for deletion (for logging)
        """
        try:
            # Validate inputs - Add this validation
            if not email_id or not isinstance(email_id, str):
                return f"Error: Invalid email_id format: {email_id}"
            if not reason or not isinstance(reason, str):
                return f"Error: Invalid reason format: {reason}"
            
            mail = self._connect()
            try:
                mail.select("INBOX")
                
                # First verify the email exists and get its details for logging
                result, data = mail.fetch(email_id, "(RFC822)")
                if result != "OK" or not data or data[0] is None:
                    return f"Error: Email with ID {email_id} not found"
                    
                msg = email.message_from_bytes(data[0][1])
                subject = decode_header_safe(msg["Subject"])
                sender = decode_header_safe(msg["From"])
                
                # Move to Trash
                mail.store(email_id, '+X-GM-LABELS', '\\Trash')
                mail.store(email_id, '-X-GM-LABELS', '\\Inbox')
                
                return f"Email deleted: '{subject}' from {sender}. Reason: {reason}"
            except Exception as e:
                return f"Error deleting email: {e}"
            finally:
                self._disconnect(mail)

        except Exception as e:
            return f"Error deleting email: {str(e)}"

# -------------------------------
# Empty Trash Tool
# -------------------------------

class EmptyTrashTool(BaseTool):
    """Tool to empty Gmail trash."""
    name: str = "empty_gmail_trash"
    description: str = "Empties the Gmail trash folder to free up space"

    def _connect(self):
        """Connect to Gmail using IMAP."""
        # Get email credentials from environment
        email_address = os.environ.get('EMAIL_ADDRESS')
        app_password = os.environ.get('APP_PASSWORD')
        
        if not email_address or not app_password:
            raise ValueError("EMAIL_ADDRESS or APP_PASSWORD environment variables not set")
        
        # Connect to Gmail's IMAP server
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        print(f"Connecting to Gmail with email: {email_address[:3]}...{email_address[-10:]}")
        mail.login(email_address, app_password)
        return mail

    def _disconnect(self, mail):
        """Disconnect from Gmail."""
        try:
            mail.logout()
        except:
            pass
    
    def _run(self) -> str:
        """
        Empty the Gmail trash folder.
        """
        try:
            mail = self._connect()
            
            # Try different trash folder names (Gmail can have different naming conventions)
            trash_folders = [
                '"[Gmail]/Trash"',
                '[Gmail]/Trash',
                'Trash',
                '"[Google Mail]/Trash"',
                '[Google Mail]/Trash'
            ]
            success = False
            trash_folder_used = None
            
            for folder in trash_folders:
                try:
                    print(f"Attempting to select trash folder: {folder}")
                    result, data = mail.select(folder)
                    if result == 'OK':
                        trash_folder_used = folder
                        print(f"Successfully selected trash folder: {folder}")
                        
                        # Search for all messages in trash
                        result, data = mail.search(None, 'ALL')
                        
                        if result == 'OK':
                            email_ids = data[0].split()
                            count = len(email_ids)
                            if count == 0:
                                print("No messages found in trash.")
                                return "Trash is already empty. No messages to delete."
                            print(f"Found {count} messages in trash.")
                            
                            # Delete all messages in trash
                            for email_id in email_ids:
                                mail.store(email_id, '+FLAGS', '\\Deleted')
                            
                            # Permanently remove messages marked for deletion
                            mail.expunge()
                            success = True
                            break
                        
                except Exception as e:
                    print(f"Error accessing trash folder {folder}: {e}")
                    continue
            if success:
                return f"Successfully emptied Gmail trash folder ({trash_folder_used}). Deleted {count} messages."
            else:
                return "Could not empty trash. No trash folder found or accessible."

        except Exception as e:
            return f"Error emptying trash: {str(e)}"
        finally:
            self._disconnect(mail)

# -------------------------------
# Thread History Tool
# -------------------------------

class ThreadHistorySchema(BaseModel):
    """Schema for GetThreadHistoryTool input."""
    email_id: str = Field(..., description="Email ID to get the thread history for")
    include_attachments: bool = Field(
        default=False, 
        description="Whether to include information about attachments in the history"
    )
    max_depth: Optional[int] = Field(
        default=10, 
        description="Maximum number of emails to retrieve in the thread history",
        ge=1
    )

class GetThreadHistoryTool(GmailToolBase):
    """Tool to retrieve the complete history of an email thread."""
    name: str = "get_thread_history"
    description: str = "Gets the complete conversation history for an email thread"
    args_schema: Type[BaseModel] = ThreadHistorySchema
    
    def _run(self, email_id: str, include_attachments: bool = False, max_depth: int = 10) -> Dict[str, Any]:
        """
        Fetch the complete history of an email thread.

        Args:
            email_id: The email ID to get thread history for
            include_attachments: Whether to include attachment info
            max_depth: Maximum number of emails to retrieve

        Returns:
            A dictionary containing the thread history and metadata
        """
        mail = self._connect()
        try:
            mail.select("INBOX")
            
            # Fetch the target email first
            result, data = mail.fetch(email_id, "(RFC822)")
            if result != "OK":
                return {"error": f"Failed to fetch email with ID {email_id}"}
                
            raw_email = data[0][1]
            root_msg = email.message_from_bytes(raw_email)
            
            # Extract the Message-ID, References, and In-Reply-To
            message_id = root_msg.get("Message-ID", "").strip()
            references = root_msg.get("References", "").strip().split()
            in_reply_to = root_msg.get("In-Reply-To", "").strip()
            
            # Collect all relevant message IDs for the thread
            thread_ids = set()
            if message_id:
                thread_ids.add(message_id)
            if in_reply_to:
                thread_ids.add(in_reply_to)
            thread_ids.update(references)
            
            # Remove any empty strings
            thread_ids = {tid for tid in thread_ids if tid}
            print(f"Found {len(thread_ids)} related message IDs in thread")
            
            # Now search for all emails in this thread
            thread_messages = []
            thread_messages.append(self._process_message(root_msg, include_attachments))
            
            if thread_ids:
                # Build search query for all messages in thread
                search_terms = []
                for tid in thread_ids:
                    # Search by header field
                    search_terms.append(f'HEADER Message-ID "{tid}"')
                    search_terms.append(f'HEADER References "{tid}"')
                    search_terms.append(f'HEADER In-Reply-To "{tid}"')
                
                search_query = ' OR '.join(search_terms)
                
                # Execute the search
                result, data = mail.search(None, search_query)
                if result == "OK" and data[0]:
                    found_ids = data[0].split()
                    print(f"Found {len(found_ids)} emails in thread")
                    
                    # Limit to max_depth
                    if len(found_ids) > max_depth:
                        found_ids = found_ids[:max_depth]
                    
                    # Process each message
                    for msg_id in found_ids:
                        # Skip the root message as we already processed it
                        if msg_id.decode('utf-8') == email_id:
                            continue
                        result, msg_data = mail.fetch(msg_id, "(RFC822)")
                        if result == "OK":
                            msg = email.message_from_bytes(msg_data[0][1])
                            thread_messages.append(self._process_message(msg, include_attachments))
            
            # Sort messages by date
            thread_messages.sort(key=lambda x: email.utils.parsedate_to_datetime(x['date']) 
                               if x.get('date') and email.utils.parsedate_to_datetime(x['date']) 
                               else datetime.datetime.min)
            
            # Create a dictionary with thread info and messages
            thread_history = {
                "thread_messages": thread_messages,
                "message_count": len(thread_messages),
                "thread_subject": decode_header_safe(root_msg["Subject"]),
                "latest_message_id": message_id,
                "participants": self._extract_participants(thread_messages)
            }
            return thread_history
            
        except Exception as e:
            print(f"Error retrieving thread history: {e}")
            traceback.print_exc()
            return {"error": str(e)}
        finally:
            self._disconnect(mail)

    def _process_message(self, msg, include_attachments: bool) -> Dict[str, Any]:
        """
        Process an email message into a structured format.
        """
        # Extract headers
        subject = decode_header_safe(msg["Subject"])
        sender = decode_header_safe(msg["From"])
        to = decode_header_safe(msg["To"])
        date = msg["Date"]
        message_id = msg.get("Message-ID", "")
        in_reply_to = msg.get("In-Reply-To", "")
        references = msg.get("References", "")
        
        # Extract body
        body = self._extract_body(msg)
        
        # Process attachments if requested
        attachments = []
        if include_attachments and msg.is_multipart():
            for part in msg.walk():
                if part.get_content_disposition() == 'attachment':
                    filename = part.get_filename()
                    if filename:
                        attachments.append({
                            "filename": decode_header_safe(filename),
                            "type": part.get_content_type(),
                            "size": len(part.get_payload(decode=True)) if part.get_payload(decode=True) else 0
                        })
        
        # Create message dict
        message_dict = {
            "subject": subject,
            "sender": sender,
            "to": to,
            "date": date,
            "body": body,
            "message_id": message_id,
            "in_reply_to": in_reply_to,
            "references": references
        }
        if include_attachments:
            message_dict["attachments"] = attachments
        return message_dict

    def _extract_participants(self, messages: List[Dict[str, Any]]) -> List[str]:
        """
        Extract all unique participants from the thread.
        """
        participants = set()
        for msg in messages:
            # Extract email addresses from sender and recipient fields
            for field in ['sender', 'to', 'cc', 'bcc']:
                if field in msg and msg[field]:
                    # Use regex to extract email addresses
                    emails = re.findall(r'[\w\.-]+@[\w\.-]+', msg[field])
                    participants.update(emails)
        return list(participants)

# -------------------------------
# Context-Aware Reply Tool
# -------------------------------

class ContextAwareReplySchema(BaseModel):
    """Schema for ContextAwareReplyTool input."""
    email_id: str = Field(..., description="Email ID to reply to")
    subject: Optional[str] = Field(None, description="Optional subject override")
    body: str = Field(..., description="Reply body content")
    include_history: bool = Field(default=True, description="Whether to include conversation history when drafting")
    max_history_depth: int = Field(default=5, description="Maximum number of emails to include in history")

class ContextAwareReplyTool(GmailToolBase):
    """Tool to draft replies with full conversation context."""
    name: str = "draft_contextual_reply"
    description: str = "Drafts a reply with full conversation context"
    args_schema: Type[BaseModel] = ContextAwareReplySchema
    
    def _run(self, email_id: str, body: str, subject: Optional[str] = None, 
             include_history: bool = True, max_history_depth: int = 5) -> str:
        """
        Draft a reply to an email with full conversation context.
        """
        mail = self._connect()
        try:
            # Fetch the email to reply to
            result, data = mail.fetch(email_id, "(RFC822)")
            if result != "OK":
                return f"Failed to fetch email with ID {email_id}"
                
            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)
            
            # Extract key information
            original_subject = decode_header_safe(msg["Subject"])
            sender = decode_header_safe(msg["From"])
            reply_to = msg.get("Reply-To", sender)
            
            # Extract email address from the sender/reply-to field
            recipient_match = re.search(r'<([^>]+)>', reply_to)
            if recipient_match:
                recipient = recipient_match.group(1)
            else:
                recipient = reply_to
            
            # Set subject (add Re: if needed)
            if subject is None or subject.strip() == "":
                if not original_subject or original_subject.strip() == "":
                    subject = "Re: No Subject"
                elif not original_subject.lower().startswith('re:'):
                    subject = f"Re: {original_subject}"
                else:
                    subject = original_subject
            
            # Get thread history if requested
            thread_info = {
                'message_id': msg.get('Message-ID', ''),
                'in_reply_to': msg.get('In-Reply-To', ''),
                'references': msg.get('References', ''),
                'email_id': email_id
            }
            
            if include_history:
                # Fetch thread history
                thread_history_tool = GetThreadHistoryTool()
                thread_history = thread_history_tool._run(
                    email_id=email_id,
                    include_attachments=True,
                    max_depth=max_history_depth
                )
                
                # Add thread history to thread_info
                thread_info['thread_messages'] = thread_history.get('thread_messages', [])
                thread_info['thread_subject'] = thread_history.get('thread_subject', '')
                thread_info['participants'] = thread_history.get('participants', [])
            
            # Use SaveDraftTool to save the draft
            draft_tool = SaveDraftTool()
            result = draft_tool._run(
                subject=subject,
                body=body,
                recipient=recipient,
                thread_info=thread_info
            )
            
            return result
            
        except Exception as e:
            print(f"Error creating contextual reply: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"
        finally:
            self._disconnect(mail)

# -------------------------------
# Email Archive Tool for Advanced Search
# -------------------------------

class EmailArchiveSchema(BaseModel):
    """Schema for EmailArchiveTool input."""
    date_from: Optional[str] = Field(None, description="Start date for filtering emails (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="End date for filtering emails (YYYY-MM-DD)")
    search_query: Optional[str] = Field(None, description="Search term to filter emails by subject, sender, or body")
    limit: int = Field(50, description="Maximum number of emails to fetch", ge=1, le=500)
    folder: str = Field("INBOX", description="Gmail folder to search in (INBOX, SENT, ALL, etc.)")
    include_read: bool = Field(True, description="Include read emails in results")
    include_unread: bool = Field(True, description="Include unread emails in results")
    sort_by: str = Field("date", description="Sort emails by: date, sender, subject")
    sort_order: str = Field("desc", description="Sort order: desc (newest first) or asc (oldest first)")

class EmailArchiveTool(GmailToolBase):
    """Tool to fetch and search through all Gmail emails with advanced filtering."""
    name: str = "email_archive_search"
    description: str = "Search and retrieve emails from Gmail with date range, search terms, and filtering options"
    args_schema: type[BaseModel] = EmailArchiveSchema

    def _run(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 50,
        folder: str = "INBOX",
        include_read: bool = True,
        include_unread: bool = True,
        sort_by: str = "date",
        sort_order: str = "desc"
    ) -> List[Tuple[str, str, str, str, Dict[str, Any]]]:
        """
        Fetch emails with advanced filtering options.
        
        Returns:
            List of tuples: (subject, sender, body, email_id, thread_info)
        """
        try:
            mail = self._connect()
            
            # Select the appropriate folder
            if folder.upper() == "ALL":
                mail.select('"[Gmail]/All Mail"')
            elif folder.upper() == "SENT":
                mail.select('"[Gmail]/Sent Mail"')
            elif folder.upper() == "DRAFTS":
                mail.select('"[Gmail]/Drafts"')
            elif folder.upper() == "TRASH":
                mail.select('"[Gmail]/Trash"')
            else:
                mail.select("INBOX")
            
            # Build search criteria
            search_criteria = []
            
            # Date filtering
            if date_from:
                try:
                    from_date = datetime.datetime.strptime(date_from, "%Y-%m-%d")
                    search_criteria.append(f'SINCE "{from_date.strftime("%d-%b-%Y")}"')
                except ValueError:
                    print(f"Invalid date_from format: {date_from}. Expected YYYY-MM-DD")
            
            if date_to:
                try:
                    to_date = datetime.datetime.strptime(date_to, "%Y-%m-%d")
                    search_criteria.append(f'BEFORE "{to_date.strftime("%d-%b-%Y")}"')
                except ValueError:
                    print(f"Invalid date_to format: {date_to}. Expected YYYY-MM-DD")
            
            # Read/Unread filtering
            if include_read and not include_unread:
                search_criteria.append("SEEN")
            elif include_unread and not include_read:
                search_criteria.append("UNSEEN")
            # If both are True, no filter needed
            
            # Text search (subject, sender, or body)
            if search_query:
                # Search in subject, from, or body
                search_terms = [
                    f'SUBJECT "{search_query}"',
                    f'FROM "{search_query}"',
                    f'BODY "{search_query}"'
                ]
                search_criteria.append(f'OR ({") OR (".join(search_terms)})')
            
            # Combine search criteria
            if search_criteria:
                search_string = " ".join(search_criteria)
            else:
                search_string = "ALL"
            
            print(f"Searching with criteria: {search_string}")
            
            # Search for emails
            status, messages = mail.search(None, search_string)
            if status != "OK":
                return []
            
            email_ids = messages[0].split()
            
            # Sort email IDs based on sort_order
            if sort_order.lower() == "asc":
                email_ids = email_ids[:limit]  # Oldest first
            else:
                email_ids = email_ids[-limit:]  # Newest first
                email_ids.reverse()  # Reverse to get newest first in the list
            
            print(f"Found {len(email_ids)} emails matching criteria, processing {min(len(email_ids), limit)}...")
            
            emails = []
            for email_id in email_ids[:limit]:
                try:
                    status, msg_data = mail.fetch(email_id, "(RFC822)")
                    if status != "OK":
                        continue
                    
                    email_body = msg_data[0][1]
                    email_message = email.message_from_bytes(email_body)
                    
                    # Extract email details
                    subject = decode_header_safe(email_message.get("subject", "No Subject"))
                    sender = decode_header_safe(email_message.get("from", "Unknown"))
                    date_header = email_message.get("date", "")
                    message_id = email_message.get("message-id", "")
                    
                    # Parse and format date
                    email_date = ""
                    try:
                        if date_header:
                            parsed_date = email.utils.parsedate_tz(date_header)
                            if parsed_date:
                                dt = datetime.datetime(*parsed_date[:6])
                                email_date = dt.strftime("%Y-%m-%d")
                    except:
                        pass
                    
                    # Extract body
                    body = ""
                    if email_message.is_multipart():
                        for part in email_message.walk():
                            if part.get_content_type() == "text/plain":
                                try:
                                    body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                                    break
                                except:
                                    continue
                            elif part.get_content_type() == "text/html" and not body:
                                try:
                                    html_body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                                    body = clean_email_body(html_body)
                                except:
                                    continue
                    else:
                        try:
                            body = email_message.get_payload(decode=True).decode("utf-8", errors="replace")
                        except:
                            body = str(email_message.get_payload())
                    
                    # Clean the body
                    body = clean_email_body(body)
                    
                    # Get thread information
                    thread_info = {
                        "date": email_date,
                        "message_id": message_id,
                        "thread_size": 1,  # Individual email, could be enhanced to get actual thread size
                        "references": email_message.get("references", ""),
                        "in_reply_to": email_message.get("in-reply-to", ""),
                        "email_id": email_id.decode() if isinstance(email_id, bytes) else str(email_id)
                    }
                    
                    emails.append((subject, sender, body, str(email_id.decode() if isinstance(email_id, bytes) else email_id), thread_info))
                    
                except Exception as e:
                    print(f"Error processing email {email_id}: {e}")
                    continue
            
            # Sort emails based on sort_by parameter
            if sort_by.lower() == "sender":
                emails.sort(key=lambda x: (x[1] or "").lower(), reverse=(sort_order.lower() == "desc"))
            elif sort_by.lower() == "subject":
                emails.sort(key=lambda x: (x[0] or "").lower(), reverse=(sort_order.lower() == "desc"))
            elif sort_by.lower() == "date":
                emails.sort(key=lambda x: x[4].get("date", ""), reverse=(sort_order.lower() == "desc"))
            
            self._disconnect(mail)
            print(f"Successfully retrieved {len(emails)} emails from {folder}")
            return emails
            
        except Exception as e:
            print(f"Error in EmailArchiveTool: {e}")
            traceback.print_exc()
            return []


# -------------------------------
# Real-Time Email Listener
# -------------------------------

def decode_header_safe(header_value):
    """Safely decode email header."""
    if not header_value:
        return ""
    
    try:
        if isinstance(header_value, str):
            return header_value
        
        decoded = decode_header(header_value)
        result = ""
        for text, encoding in decoded:
            if isinstance(text, bytes):
                if encoding:
                    result += text.decode(encoding, errors='ignore')
                else:
                    result += text.decode('utf-8', errors='ignore')
            else:
                result += str(text)
        return result
    except Exception:
        return str(header_value)


class EmailListener:
    """Real-time email listener using IMAP IDLE."""
    
    def __init__(self, callback: Callable[[List[Tuple]], None] = None, email_address: str = None, app_password: str = None):
        """
        Initialize the email listener.
        
        Args:
            callback: Function to call when new emails are detected.
                     Should accept a list of email tuples.
            email_address: Gmail email address (optional, will try environment if not provided)
            app_password: Gmail app password (optional, will try environment if not provided)
        """
        # Use provided credentials or fall back to environment variables
        self.email_address = email_address or os.environ.get("EMAIL_ADDRESS")
        self.app_password = app_password or os.environ.get("APP_PASSWORD")
        self.imap_server = os.environ.get("IMAP_SERVER", "imap.gmail.com")
        self.imap_port = int(os.environ.get("IMAP_PORT", "993"))
        
        # Initialize attributes even if credentials are missing
        self.callback = callback
        self.is_listening = False
        self.mail = None
        self.listener_thread = None
        self.stop_event = threading.Event()
        
        # Activity logs for dashboard
        self.activity_logs = []
        
        # Track monitoring start time and processed emails to avoid reprocessing old emails
        self.monitoring_start_time = None
        self.processed_email_ids = set()  # Track processed email IDs to avoid duplicates
        
        # Statistics
        self.stats = {
            "start_time": None,
            "emails_detected": 0,
            "connection_errors": 0,
            "last_activity": None,
            "status": "stopped"
        }
        
        # Check credentials but don't fail initialization
        if not self.email_address or not self.app_password:
            self._log_activity("⚠️ No credentials provided - please call set_credentials() or provide EMAIL_ADDRESS/APP_PASSWORD", "warning")
            self.stats["status"] = "configuration_error"

    def set_credentials(self, email_address: str, app_password: str):
        """
        Set Gmail credentials for the listener.
        
        Args:
            email_address: Gmail email address
            app_password: Gmail app password
        """
        self.email_address = email_address
        self.app_password = app_password
        
        # Update status if credentials are now available
        if self.email_address and self.app_password:
            if self.stats["status"] == "configuration_error":
                self.stats["status"] = "stopped"
            self._log_activity(f"✅ Credentials updated for {email_address[:3]}***@gmail.com", "success")
        else:
            self._log_activity("❌ Invalid credentials provided", "error")
            self.stats["status"] = "configuration_error"

    def _log_activity(self, message: str, level: str = "info"):
        """Log activity with timestamp for dashboard display."""
        timestamp = dt.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "message": message,
            "level": level
        }
        self.activity_logs.append(log_entry)
        
        # Keep only last 100 entries
        if len(self.activity_logs) > 100:
            self.activity_logs = self.activity_logs[-100:]
        
        # Also print for console output
        level_icons = {
            "info": "ℹ️",
            "success": "✅", 
            "warning": "⚠️",
            "error": "❌"
        }
        icon = level_icons.get(level, "ℹ️")
        print(f"{icon} {message}")

    def _connect(self) -> imaplib.IMAP4_SSL:
        """Establish IMAP connection."""
        try:
            self._log_activity(f"Connecting to {self.imap_server}:{self.imap_port}")
            mail = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            mail.login(self.email_address, self.app_password)
            mail.select("INBOX")
            self._log_activity("Successfully connected to Gmail IMAP", "success")
            return mail
        except Exception as e:
            self._log_activity(f"Error connecting to IMAP: {e}", "error")
            self.stats["connection_errors"] += 1
            raise

    def _disconnect(self, mail: imaplib.IMAP4_SSL):
        """Disconnect from IMAP."""
        try:
            if mail:
                # Check if connection is still alive and in a valid state
                try:
                    # Try to close the current mailbox if one is selected
                    mail.close()
                except (imaplib.IMAP4.error, ConnectionResetError, OSError):
                    # Ignore errors if already closed or in wrong state
                    pass
                
                try:
                    # Try to logout
                    mail.logout()
                except (imaplib.IMAP4.error, ConnectionResetError, OSError):
                    # Ignore errors if already logged out or connection lost
                    pass
            self._log_activity("Disconnected from Gmail IMAP")
        except Exception as e:
            # Log as info instead of warning since disconnection errors are common
            self._log_activity(f"Disconnect completed (with minor cleanup issues): {e}", "info")

    def _check_for_new_emails(self) -> List[Tuple]:
        """Check for ONLY NEW emails that arrived after monitoring started."""
        try:
            if not self.monitoring_start_time:
                self._log_activity("⚠️ No monitoring start time set, cannot filter new emails", "warning")
                return []
            
            # Use existing connection (should be fresh from polling loop)
            if not self.mail:
                self._log_activity("❌ No IMAP connection available for checking emails", "error")
                return []
            
            # Ensure we're in the right folder
            try:
                self.mail.select("INBOX")
            except Exception as e:
                self._log_activity(f"❌ Error selecting INBOX: {e}", "error")
                return []
            
            # Search for unread emails received after monitoring started
            # Format time for IMAP search (DD-MMM-YYYY format)
            since_date = self.monitoring_start_time.strftime("%d-%b-%Y")
            
            # Search for unread emails since monitoring started
            result, data = self.mail.search(None, f'(UNSEEN SINCE {since_date})')
            
            if result != "OK":
                self._log_activity(f"❌ Error searching for new emails: {result}", "error")
                return []
            
            email_ids = data[0].split() if data[0] else []
            if not email_ids:
                return []
            
            # Get newest emails first (reverse order)
            email_ids = list(reversed(email_ids))
            
            new_emails = []
            processed_count = 0
            
            for email_id in email_ids[:5]:  # Limit to 5 most recent
                email_id_str = email_id.decode() if isinstance(email_id, bytes) else str(email_id)
                
                # Skip if already processed
                if email_id_str in self.processed_email_ids:
                    continue
                
                try:
                    # Fetch email details
                    result, msg_data = self.mail.fetch(email_id, "(RFC822)")
                    if result != "OK":
                        continue
                    
                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email)
                    
                    # Parse email date and compare with monitoring start time
                    date_str = msg.get("Date", "")
                    if date_str:
                        try:
                            email_date = email.utils.parsedate_tz(date_str)
                            if email_date:
                                email_datetime = dt.fromtimestamp(email.utils.mktime_tz(email_date))
                                
                                # Only process emails received AFTER monitoring started
                                if email_datetime <= self.monitoring_start_time:
                                    self._log_activity(f"⏭️ Skipping old email from {email_datetime.strftime('%H:%M:%S')} (before monitoring)", "info")
                                    continue
                        except Exception as e:
                            self._log_activity(f"⚠️ Could not parse email date: {e}", "warning")
                            # If we can't parse date, be conservative and skip
                            continue
                    
                    # Process the email
                    subject = decode_header_safe(msg.get("Subject", "No Subject"))
                    sender = decode_header_safe(msg.get("From", "Unknown"))
                    body = self._extract_body(msg)
                    
                    # Create thread info
                    thread_info = {
                        "email_id": email_id_str,
                        "date": email_datetime.strftime("%Y-%m-%d") if 'email_datetime' in locals() else "",
                        "thread_size": 1
                    }
                    
                    email_tuple = (subject, sender, body, email_id_str, thread_info)
                    new_emails.append(email_tuple)
                    
                    # Mark as processed to avoid reprocessing
                    self.processed_email_ids.add(email_id_str)
                    processed_count += 1
                    
                    self._log_activity(f"📧 New email detected: '{subject[:50]}...' from {sender[:30]}", "info")
                    
                except Exception as e:
                    self._log_activity(f"❌ Error processing email {email_id_str}: {e}", "error")
                    continue
            
            if new_emails:
                self._log_activity(f"✅ Found {len(new_emails)} genuinely NEW emails (received after monitoring started)", "success")
            else:
                self._log_activity("ℹ️ No new emails found since monitoring started", "info")
            
            return new_emails
            
        except Exception as e:
            self._log_activity(f"❌ Error checking for new emails: {e}", "error")
            traceback.print_exc()
            return []

    def _extract_body(self, msg) -> str:
        """Extract email body text."""
        try:
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode('utf-8', errors='ignore')
                            break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode('utf-8', errors='ignore')
            
            # Clean up the body
            if body:
                body = re.sub(r'\r\n', '\n', body)
                body = re.sub(r'\n+', '\n', body)
                body = body.strip()
            
            return body[:1000]  # Limit body length
        except Exception as e:
            return f"Error extracting body: {e}"

    def _idle_loop(self):
        """Main IDLE loop for listening to new emails."""
        self.stats["start_time"] = dt.now().isoformat()
        self.stats["status"] = "starting"
        self._log_activity("Starting real-time email monitoring")
        
        # Use polling approach for reliability
        self._polling_loop()

    def _polling_loop(self):
        """Polling loop for email detection - more reliable than IDLE."""
        self._log_activity("Starting email polling mode (10 second interval)", "info")
        last_message_count = 0
        last_check_time = dt.now()
        
        while not self.stop_event.is_set():
            try:
                # Always create a fresh connection for each poll cycle to avoid state issues
                if self.mail:
                    try:
                        self._disconnect(self.mail)
                    except:
                        pass
                    self.mail = None
                
                # Connect fresh for each poll
                self.mail = self._connect()
                self.stats["status"] = "listening"
                
                # Select INBOX and get current message count
                status, messages = self.mail.search(None, 'ALL')
                
                if status == 'OK':
                    current_count = len(messages[0].split()) if messages[0] else 0
                    current_time = dt.now()
                    
                    # Check if there are new messages since last check
                    if current_count > last_message_count:
                        self._log_activity(f"📧 New emails detected! Total count: {current_count} (was {last_message_count})", "info")
                        
                        # Get only unread emails (which should be the new ones)
                        new_emails = self._check_for_new_emails()
                        
                        if new_emails:
                            self.stats["emails_detected"] += len(new_emails)
                            self.stats["last_activity"] = current_time.isoformat()
                            
                            self._log_activity(f"📧 Processing {len(new_emails)} new unread emails...", "info")
                            
                            # Call callback if provided
                            if self.callback:
                                try:
                                    self.callback(new_emails)
                                    self._log_activity(f"✅ Successfully processed {len(new_emails)} emails through AI", "success")
                                except Exception as e:
                                    self._log_activity(f"❌ Error in callback: {e}", "error")
                                    traceback.print_exc()
                        else:
                            self._log_activity("ℹ️ No new unread emails found despite count increase", "warning")
                    
                    last_message_count = current_count
                    last_check_time = current_time
                
                # Clean disconnect after each poll cycle
                if self.mail:
                    try:
                        self._disconnect(self.mail)
                    except:
                        pass
                    self.mail = None
                
                # Wait before next poll (10 seconds for responsiveness)
                if not self.stop_event.wait(10):
                    continue
                else:
                    break
                    
            except Exception as e:
                self._log_activity(f"❌ Error in polling loop: {e}", "error")
                self.stats["connection_errors"] += 1
                self.stats["status"] = "error"
                
                # Clean up connection on error
                if self.mail:
                    try:
                        self._disconnect(self.mail)
                    except:
                        pass
                    self.mail = None
                
                # Wait before retrying
                if not self.stop_event.wait(30):
                    self._log_activity("🔄 Retrying connection after error...", "info")
                    continue
                else:
                    break
        
        # Final cleanup on exit
        if self.mail:
            try:
                self._disconnect(self.mail)
            except:
                pass
            self.mail = None
        self.stats["status"] = "stopped"
        self._log_activity("📪 Email listener stopped", "info")

    def start_listening(self):
        """Start the email listener in a background thread."""
        if self.is_listening:
            self._log_activity("Email listener is already running", "warning")
            return False
        
        # Check for required credentials
        if not self.email_address or not self.app_password:
            self._log_activity("Error: EMAIL_ADDRESS and APP_PASSWORD must be set in environment", "error")
            self.stats["status"] = "configuration_error"
            return False
            
        # Set monitoring start time to track only NEW emails
        self.monitoring_start_time = dt.now()
        self.processed_email_ids.clear()  # Clear any previous processed emails
        
        self.is_listening = True
        self.stop_event.clear()
        self.listener_thread = threading.Thread(target=self._idle_loop, daemon=True)
        self.listener_thread.start()
        self._log_activity(f"✅ Email listener started - monitoring emails received after {self.monitoring_start_time.strftime('%H:%M:%S')}", "success")
        return True

    def stop_listening(self):
        """Stop the email listener."""
        if not self.is_listening:
            self._log_activity("Email listener is not running", "warning")
            return False
            
        self.is_listening = False
        self.stop_event.set()
        self._log_activity("🛑 Stopping email listener...", "info")
        
        # Wait for thread to finish
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=5)
            
        self._log_activity("✅ Email listener stopped successfully", "success")
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current listener status and statistics."""
        # Ensure stats is always properly initialized
        if not hasattr(self, 'stats') or self.stats is None:
            self.stats = {
                "start_time": None,
                "emails_detected": 0,
                "connection_errors": 0,
                "last_activity": None,
                "status": "stopped"
            }
        
        # Ensure is_listening is properly initialized
        if not hasattr(self, 'is_listening'):
            self.is_listening = False
            
        # Ensure listener_thread is properly initialized
        if not hasattr(self, 'listener_thread'):
            self.listener_thread = None
        
        return {
            "is_listening": self.is_listening,
            "stats": self.stats.copy(),
            "thread_alive": self.listener_thread.is_alive() if self.listener_thread else False
        }


# -------------------------------
# Email Processor for Real-time Processing
# -------------------------------

class RealTimeEmailProcessor:
    """Processes emails in real-time using CrewAI."""
    
    def __init__(self):
        self.processing_queue = []
        self.is_processing = False
        self.processed_emails = []  # Track all processed emails
        self.generated_responses = []  # Track all generated responses
        
        # Initialize stats tracking
        self.stats = {
            "total_processed": 0,
            "total_responses_generated": 0,
            "last_processed": None,
            "start_time": dt.now().isoformat(),
            "processing_errors": 0
        }
        
    def get_processed_emails(self) -> List[Dict[str, Any]]:
        """Get all processed emails from real-time monitoring."""
        return self.processed_emails.copy()
    
    def get_generated_responses(self) -> List[Dict[str, Any]]:
        """Get all generated responses from real-time monitoring."""
        return self.generated_responses.copy()
    
    def clear_history(self):
        """Clear processed emails and responses history."""
        self.processed_emails.clear()
        self.generated_responses.clear()
        print("🧹 Cleared real-time monitoring history")
        
    def process_new_emails(self, emails: List[Tuple]):
        """Process new emails detected by the listener."""
        if not emails:
            return
            
        print(f"🎯 Processing {len(emails)} new emails...")
        processing_timestamp = dt.now().isoformat()
        
        try:
            # Save emails to fetched_emails.json for CrewAI processing
            os.makedirs("output", exist_ok=True)
            
            # Convert email tuples to EmailDetails format
            try:
                from gmail_crew_ai.models import EmailDetails
            except ImportError:
                # Fallback: create a simple email dict if models aren't available
                print("⚠️ EmailDetails model not found, using simple dict format")
                self._process_emails_simple_format(emails, processing_timestamp)
                return
            
            from datetime import date
            
            email_details = []
            today = date.today()
            
            for i, email_tuple in enumerate(emails):
                try:
                    print(f"📧 Processing email {i+1}/{len(emails)}")
                    email_detail = EmailDetails.from_email_tuple(email_tuple)
                    
                    # Calculate age if date is available
                    if email_detail.date:
                        try:
                            email_date_obj = dt.strptime(email_detail.date, "%Y-%m-%d").date()
                            email_detail.age_days = (today - email_date_obj).days
                        except Exception as e:
                            print(f"⚠️ Error calculating age for email date {email_detail.date}: {e}")
                            email_detail.age_days = None
                    
                    email_dict = email_detail.dict()
                    email_details.append(email_dict)
                    
                    # Add to processed emails history with timestamp
                    processed_email = {
                        **email_dict,
                        "processed_at": processing_timestamp,
                        "processing_method": "real_time_monitoring",
                        "status": "processed"
                    }
                    self.processed_emails.append(processed_email)
                    
                    print(f"✅ Processed: {email_detail.subject[:50]}... from {email_detail.sender}")
                except Exception as e:
                    print(f"❌ Error processing email {i+1}: {e}")
                    traceback.print_exc()
                    continue
            
            # Instead of appending to existing emails, replace them with new ones
            # This ensures CrewAI always processes the latest emails
            with open('output/fetched_emails.json', 'w') as f:
                json.dump(email_details, f, indent=2)
            
            print(f"💾 Saved {len(email_details)} new emails to output/fetched_emails.json")
            
            # Small delay to ensure file is written
            time.sleep(0.5)
            
            # Run CrewAI processing and track responses
            self._run_crew_processing_with_tracking(len(email_details), processing_timestamp)
            
        except Exception as e:
            print(f"❌ Error processing new emails: {e}")
            traceback.print_exc()
    
    def _process_emails_simple_format(self, emails: List[Tuple], processing_timestamp: str):
        """Fallback method to process emails in simple format."""
        try:
            email_details = []
            for i, email_tuple in enumerate(emails):
                subject, sender, body, email_id, thread_info = email_tuple
                email_dict = {
                    "subject": subject,
                    "sender": sender, 
                    "body": body,
                    "email_id": email_id,
                    "thread_info": thread_info,
                    "date": thread_info.get('date', '') if thread_info else '',
                    "age_days": None
                }
                email_details.append(email_dict)
                
                # Add to processed emails history
                processed_email = {
                    **email_dict,
                    "processed_at": processing_timestamp,
                    "processing_method": "real_time_monitoring",
                    "status": "processed"
                }
                self.processed_emails.append(processed_email)
                
                print(f"✅ Simple format - processed: {subject[:50]}... from {sender}")
            
            with open('output/fetched_emails.json', 'w') as f:
                json.dump(email_details, f, indent=2)
            
            print(f"💾 Saved {len(email_details)} emails using simple format")
            self._run_crew_processing_with_tracking(len(email_details), processing_timestamp)
            
        except Exception as e:
            print(f"❌ Error in simple format processing: {e}")
            traceback.print_exc()
            
        except Exception as e:
            print(f"Error processing new emails: {e}")
            traceback.print_exc()
    
    def _run_crew_processing_with_tracking(self, email_count: int, processing_timestamp: str):
        """Run CrewAI processing and track generated responses."""
        try:
            print(f"🤖 Starting CrewAI processing for {email_count} emails...")
            
            # Import and run the crew
            from gmail_crew_ai.crew import GmailCrewAi
            
            crew = GmailCrewAi()
            
            # Create inputs that tell CrewAI not to fetch emails again
            inputs = {
                'email_limit': email_count,
                'user_email': os.environ.get('EMAIL_ADDRESS', 'user@gmail.com'),
                'skip_email_fetch': True  # Signal to skip fetching
            }
            
            result = crew.crew().kickoff(inputs=inputs)
            
            print(f"✅ CrewAI processing completed!")
            print(f"📝 Result: {str(result)[:200]}..." if len(str(result)) > 200 else f"📝 Result: {result}")
            
            # Try to track the generated response
            try:
                # Check if response_report.json was created/updated
                response_file = 'output/response_report.json'
                if os.path.exists(response_file):
                    with open(response_file, 'r') as f:
                        response_data = json.load(f)
                    
                    # Add to generated responses history
                    tracked_response = {
                        "response_data": response_data,
                        "generated_at": processing_timestamp,
                        "email_count": email_count,
                        "processing_method": "real_time_monitoring",
                        "crew_result": str(result)[:500]  # Truncate long results
                    }
                    self.generated_responses.append(tracked_response)
                    print(f"📊 Tracked response for {email_count} emails")
                    
                else:
                    print("⚠️ No response_report.json found after CrewAI processing")
                    
            except Exception as e:
                print(f"⚠️ Error tracking response: {e}")
            
            # Update stats
            self.stats["total_processed"] += email_count
            self.stats["last_processed"] = processing_timestamp
            
            return result
            
        except Exception as e:
            print(f"❌ Error running CrewAI processing: {e}")
            traceback.print_exc()

    def _run_crew_processing(self, email_count: int):
        """Run CrewAI processing on new emails."""
        try:
            from gmail_crew_ai.crew import GmailCrewAi
            
            print(f"🤖 Running CrewAI processing for {email_count} emails...")
            crew = GmailCrewAi()
            
            # Create inputs that tell CrewAI not to fetch emails again
            inputs = {
                'email_limit': email_count,
                'user_email': os.environ.get('EMAIL_ADDRESS', 'user@gmail.com'),
                'skip_email_fetch': True  # Signal to skip fetching
            }
            
            result = crew.crew().kickoff(inputs=inputs)
            
            print("✅ CrewAI processing completed successfully")
            
            # Log the result for debugging
            if hasattr(result, 'raw') and result.raw:
                print(f"🎯 CrewAI Result Summary: {str(result.raw)[:200]}...")
            
            return result
            
        except Exception as e:
            print(f"❌ Error running CrewAI processing: {e}")
            traceback.print_exc()
            return None


# Global listener instance
_email_listener = None

def get_email_listener() -> EmailListener:
    """Get or create the global email listener instance."""
    global _email_listener
    if _email_listener is None:
        try:
            processor = RealTimeEmailProcessor()
            _email_listener = EmailListener(callback=processor.process_new_emails)
            # Store a reference to the processor for API access
            _email_listener.email_processor = processor
            print("Email listener instance created successfully")
        except Exception as e:
            print(f"Error creating email listener: {e}")
            # Create a basic listener without callback as fallback
            _email_listener = EmailListener()
    return _email_listener