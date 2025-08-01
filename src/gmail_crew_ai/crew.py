from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from crewai_tools import FileReadTool
import json
import os
from typing import List, Dict, Any, Callable
from pydantic import SkipValidation
from datetime import date, datetime

from gmail_crew_ai.tools.gmail_tools import GetUnreadEmailsTool, SaveDraftTool
from gmail_crew_ai.models import EmailResponse, EmailDetails, EmailResponseList

@CrewBase
class GmailCrewAi():
	"""Crew that processes emails."""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@before_kickoff
	def fetch_emails(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
		"""Fetch emails before starting the crew and calculate ages."""
		
		# Check if we should skip email fetching (for real-time processing)
		skip_fetch = inputs.get('skip_email_fetch', False)
		if skip_fetch:
			print("⏭️ Skipping email fetch - using existing emails from real-time listener")
			return inputs
		
		print("Fetching emails before starting the crew...")
		
		# Get the email limit from inputs
		email_limit = inputs.get('email_limit', 5)
		user_email = inputs.get('user_email', 'user@gmail.com')
		print(f"Fetching {email_limit} emails for user: {user_email}")
		
		# Create the output directory if it doesn't exist
		os.makedirs("output", exist_ok=True)
		
		# Use the GetUnreadEmailsTool directly - get latest emails first, then prioritize
		try:
			email_tool = GetUnreadEmailsTool()
			# Get the latest emails without deep prioritization scanning (like Gmail's indexed approach)
			email_tuples = email_tool._run(limit=email_limit, prioritize_primary=False)
			
			# Now apply smart prioritization (like Gmail's ML-based categorization)
			if email_tuples:
				def calculate_email_priority(email_tuple):
					"""Calculate priority score like Gmail's algorithm"""
					subject, sender, body, email_id, thread_info = email_tuple
					score = 100  # Base priority
					
					sender_lower = sender.lower()
					subject_lower = subject.lower()
					
					# High priority indicators (like VIP senders)
					high_priority_keywords = ['urgent', 'important', 'asap', 'deadline']
					if any(keyword in subject_lower for keyword in high_priority_keywords):
						score += 50
					
					# Promotional/automated email detection (lower priority)
					promotional_indicators = [
						'noreply', 'no-reply', 'newsletter', 'unsubscribe', 'marketing',
						'promotion', 'deal', 'offer', 'sale', 'discount', 'coupon',
						'pinterest', 'facebook', 'twitter', 'instagram', 'linkedin',
						'notifications', 'automated', 'bot', 'system',
						'updates@', 'news@', 'alerts@', 'digest'
					]
					
					if any(indicator in sender_lower for indicator in promotional_indicators):
						score -= 30  # Lower priority for promotional emails
					
					# Personal emails get higher priority (like Gmail's importance markers)
					if '@gmail.com' in sender_lower or '@outlook.com' in sender_lower:
						score += 20
					
					return score
				
				# Sort by priority score (highest first) - like Gmail's smart inbox
				email_tuples = sorted(email_tuples, key=calculate_email_priority, reverse=True)
				
				# Separate for debugging
				primary_count = sum(1 for email_tuple in email_tuples 
					if calculate_email_priority(email_tuple) >= 100)
				other_count = len(email_tuples) - primary_count
				
				print(f"DEBUG: Smart prioritization applied - High Priority: {primary_count}, Lower Priority: {other_count}")
			
			
			if not email_tuples:
				print("No unread emails found or error occurred during fetching.")
				# Create an empty file to indicate the process ran but found no emails
				with open('output/fetched_emails.json', 'w') as f:
					json.dump([], f)
				return inputs
			
			# Convert email tuples to EmailDetails objects with pre-calculated ages
			emails = []
			today = date.today()
			
			for email_tuple in email_tuples:
				try:
					email_detail = EmailDetails.from_email_tuple(email_tuple)
					
					# Calculate age if date is available
					if email_detail.date:
						try:
							email_date_obj = datetime.strptime(email_detail.date, "%Y-%m-%d").date()
							email_detail.age_days = (today - email_date_obj).days
							print(f"Email date: {email_detail.date}, age: {email_detail.age_days} days")
						except Exception as e:
							print(f"Error calculating age for email date {email_detail.date}: {e}")
							email_detail.age_days = None
					
					emails.append(email_detail.dict())
				except Exception as e:
					print(f"Error processing email: {e}")
					continue
			
			# Save emails to file
			with open('output/fetched_emails.json', 'w') as f:
				json.dump(emails, f, indent=2)
			
			print(f"Fetched and saved {len(emails)} emails to output/fetched_emails.json")
			
		except Exception as e:
			print(f"Error in fetch_emails: {e}")
			import traceback
			traceback.print_exc()
			# Create an empty file to indicate the process ran but encountered an error
			with open('output/fetched_emails.json', 'w') as f:
				json.dump([], f)
		
		return inputs
	
	@property
	def llm(self):
		"""Lazy initialization of LLM to ensure environment variables are set."""
		if not hasattr(self, '_llm'):
			model = os.getenv("MODEL")
			if not model:
				raise ValueError("MODEL environment variable not set. Please login to set model credentials.")
			
			api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
			if not api_key:
				raise ValueError("API key not set. Please login to set API credentials.")
			
			self._llm = LLM(
				model=model,
				api_key=api_key,
			)
		return self._llm

	@agent
	def response_generator(self) -> Agent:
		"""The email response generator agent."""
		return Agent(
			config=self.agents_config['response_generator'],
			tools=[SaveDraftTool(), FileReadTool()],
			llm=self.llm,
		)
	
	@task
	def response_task(self) -> Task:
		"""The email response task."""
		return Task(
			config=self.tasks_config['response_task'],
			# Use EmailResponseList instead of List[EmailResponse]
			output_pydantic=EmailResponseList,
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the email processing crew."""
		# Use self.agents and self.tasks which are populated by the decorators
		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True
		)
