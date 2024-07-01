import boto3
import botocore
import json
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
import os

# loading in environment variables
load_dotenv()
# setting default session with AWS CLI Profile
boto3.setup_default_session(profile_name=os.getenv('profile_name'))
# Setup Bedrock client
config = botocore.config.Config(connect_timeout=300, read_timeout=300)
bedrock = boto3.client('bedrock-runtime' , 'us-east-1', config = config)



def parse_xml(xml, tag):
  start_tag = f"<{tag}>"
  end_tag = f"</{tag}>"
  
  start_index = xml.find(start_tag)
  if start_index == -1:
    return ""

  end_index = xml.find(end_tag)
  if end_index == -1:
    return ""

  value = xml[start_index+len(start_tag):end_index]
  return value


def extract(pdf, file_name):
   # Open the PDF file
    text = ""
    with st.status("Processing PDF", expanded=False, state="running") as status:
        with pdfplumber.open(pdf) as pdf:
        # Loop through each page in the PDF
            for page in pdf.pages:
                # Extract the text from the page
                #concat the full text
                text = text + page.extract_text()


                # Print the extracted text
                print(text)

        #save text to streamlit session state
        if 'text' not in st.session_state:
            st.session_state['text'] = text
        
        status.update(label=":heavy_check_mark: PDF Processing Complete", state="running", expanded=False)
        st.write(":heavy_check_mark: PDF Processing Complete")

        #extract info on the parties involved
        status.update(label="Extracting Party Information", state="running", expanded=False)
        scratch, party_output, party_confidence, party_work=extract_party_info(text)
        st.write(f":heavy_check_mark: Party Information Extracted")
        json_party=json.loads(party_output)
        st.json(json_party)
        st.write(f"Confidence: {party_confidence}")
        st.write(f"Explanation: {party_work}")

        #Categorize Investment Objectives
        status.update(label="Categorizing Investment Objectives", state="running", expanded=False)
        scratch, objective_output, objective_confidence,objective_work=extract_investment_objectives(text)
        st.write(f":heavy_check_mark: Investment Objectives Categorized")
        json_objective=json.loads(objective_output)
        st.json(json_objective)
        st.write(f"Confidence: {objective_confidence}")
        st.write(f"Explanation: {objective_work}")

        #Custodian and Brokerage infromation
        status.update(label="Extracting Custodian and Brokerage Information", state="running", expanded=False)
        scratch, custodian_output, custodian_confidence, custodian_work=extract_custodian_info(text)
        st.write(f":heavy_check_mark: Custodian and Brokerage Information Extracted")
        json_custodian=json.loads(custodian_output)
        st.json(json_custodian)
        st.write(f"Confidence: {custodian_confidence}")
        st.write(f"Explanation: {custodian_work}")

        #Fee Details
        status.update(label="Extracting Fee Details", state="running", expanded=False)
        scratch, fee_output, fee_confidence, fee_work=extract_fee_info(text)
        st.write(f":heavy_check_mark: Fee Details Extracted")
        json_fee=json.loads(fee_output)
        st.json(json_fee)
        st.write(f"Confidence: {fee_confidence}")
        st.write(f"Explanation: {fee_work}")

        #Effective Date
        status.update(label="Extracting Effective Date", state="running", expanded=False)
        scratch, effective_output, effective_confidence, effective_work=extract_effective_date(text)
        st.write(f":heavy_check_mark: Effective Date Extracted")
        json_effective=json.loads(effective_output)
        st.json(json_effective)
        st.write(f"Confidence: {effective_confidence}")
        st.write(f"Explanation: {effective_work}")

        #create final json
        status.update(label="Creating Final JSON", state="running", expanded=False)
        combined_json = final_json(json_party, json_objective, json_custodian, json_fee, json_effective, file_name)
        st.write(f":heavy_check_mark: Final JSON Created")
        st.json(combined_json)

        status.update(label=":heavy_check_mark: Details Extracted", state="complete", expanded=False)

    with st.expander("Full JSON Payload"):
        st.json(combined_json)



def extract_party_info(content):

    system_prompt="""
You are an AI Data Processor whose goal is to identify and extract key information from an Investment Management Agreement for a wealth management client. You will be provided a Wealth Management Agreement (DIMA, IMA, or MDIMA)
Using the context of the provided agreement Identify and Extract the following information and return it in json format:
    Parties Involved
        Client Name (name of the person acting as or on behalf of the client - often times this will be the Signee for the Client)
        Client Firm (Client Firm name, "individual" if no associated client firm)
        Investment Manager Name (name of the person acting as or on behalf of the client - often times this will be the Signee for the Investment Firm)
        Investment Manager Firm (Investment Firm name, "individual" if no associated investment firm)

<example_format>
{
  "Parties Involved": {
    "Client Name": "(Client Name)",
    "Client Firm": "(Client Firm Name or "individual")",
    "Investment Manager Name": "(Investment Manager Name)",
    "Investment Manager Firm": "(Investment Management Firm, or "individual")"
  }
}
</example_format>

Also return your confidence level representing how confident you that you were able to accurately identify and categorize the requested information. Your confidence value should be one of "High", "Medium", or "Low"
Use High Confidence in cases you have no doubt you have identified and captured ALL the correct information and there was no ambiguity and there is no need for a human to review
Use Medium Confidence in cases where you are reasonably confident you have identified most of the information, but there might be some ambiguity and a human review would be beneficial
Use Low Confidence in cases where you are not confident in the information you have captured or the data provided didn't include enough information, and a human needs to review


Think through each step of your thought process and write your thoughts down in <scratchpad> xml tags
return the valid json array with the extracted details in <output> xml tags, only including the valid json
return your confidence level (Low, Medium, or High) in <confidence> xml tags
record your explanation for your response and cite your work and reasoning in <show_work> xml tags

"""

    
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "temperature": 0.5,
        "system": system_prompt,
        "messages": [    
            {
                "role": "user",
                "content": f"<document> {content} </document>"
            }
        ]
    }

    prompt = json.dumps(prompt)

    print("------------------------------------------------------")

    response = bedrock.invoke_model(body=prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get('body').read())
    llmOutput=response_body['content'][0]['text']

    
    scratch = parse_xml(llmOutput, "scratchpad")
    output = parse_xml(llmOutput, "output")
    confidence = parse_xml(llmOutput, "confidence")
    show_work = parse_xml(llmOutput, "show_work")

    return scratch, output, confidence, show_work



def extract_investment_objectives(content):

    system_prompt="""
You are an AI Data Processor whose goal is to identify and extract key information from an Investment Management Agreement for a wealth management client. You will be provided a Wealth Management Agreement (DIMA, IMA, or MDIMA)
Using the document context of the provided agreement Categorize the Investment Objective using valid categories provided in <valid_categories> -- do not make any assumptions when determining your categorization (if no objective is mentioned or there are reference to other documents not provided that would contain this information select "Other or Not Found" and use a "Low" confidence rating)
Return your Categorization in a valid json object using the <example_format> provided below 
If multiple categories apply, include all applicable categories in the json object

Only the below categories are valid options
<valid_categories>
Capital Preservation
Income Generation
Growth
Balanced Growth and Income
Aggressive Growth
Tax-efficient investing
Socially Responsible Investing
Retirement Planning
Education Funding
Other or Not Found
</valid_categories>

<example_format>
{
  "Investment Objectives": {
    "Objective1": "(Objective)",
    "ObjectiveX": "(Objective - X representing the next number in order, only include these if there were multiple categories identified)"
  }
}
</example_format>

Also return your confidence level representing how confident you that you were able to accurately identify and categorize the requested information. Your confidence value should be one of "High", "Medium", or "Low"
Use High Confidence in cases you have no doubt you have identified and captured ALL the correct information and there was no ambiguity and there is no need for a human to review
Use Medium Confidence in cases where you are reasonably confident you have identified most of the information, but there might be some ambiguity and a human review would be beneficial
Use Low Confidence in cases where you are not confident in the information you have captured or the data provided didn't include enough information, and a human needs to review


Think through each step of your thought process and write your thoughts down in <scratchpad> xml tags
return the valid json array with the extracted details in <output> xml tags, only including the valid json
return your confidence level (Low, Medium, or High) in <confidence> xml tags
record your explanation for your response and cite your work and reasoning in <show_work> xml tags

"""

    
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "temperature": 0.5,
        "system": system_prompt,
        "messages": [    
            {
                "role": "user",
                "content": f"<document> {content} </document>"
            }
        ]
    }

    prompt = json.dumps(prompt)

    print("------------------------------------------------------")

    response = bedrock.invoke_model(body=prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get('body').read())
    llmOutput=response_body['content'][0]['text']

    
    scratch = parse_xml(llmOutput, "scratchpad")
    output = parse_xml(llmOutput, "output")
    confidence = parse_xml(llmOutput, "confidence")
    show_work = parse_xml(llmOutput, "show_work")

    return scratch, output, confidence, show_work


def extract_custodian_info(content):

    system_prompt="""
You are an AI Data Processor whose goal is to identify and extract key information from an Investment Management Agreement for a wealth management client. You will be provided a Wealth Management Agreement (DIMA, IMA, or MDIMA)
Using the document context of the provided agreement Identify and extract (making no changes) the details of information relevant to the roles, respososbilites, and all other information pertatining to the Custodian and Brokerage agreements present in the agreement. Make sure to capture all pertinent agreement information and relevant passages
Return your Results in a valid json object using the <example_format> provided below

<custodian_example>
"1. The assets comprising the Client's account shall be held by ABC Trust Company, a nationally recognized custodian, or such other custodian as the parties may mutually agree upon in writing." 
"2. XYZ Custodial Services, LLC, shall act as the custodian for the Client's account and shall provide custody and safekeeping services for the assets in the account." 
"3. The Client's assets shall be held in a separate account at DEF Bank, a member of the Federal Deposit Insurance Corporation (FDIC)."
</custodian_example>


<brokerage_example>
"1. All transactions for the Client's account shall be executed through GHI Brokerage Firm, a registered broker-dealer and member of the Financial Industry Regulatory Authority (FINRA)." 
"2. The Investment Manager is authorized to select and utilize various broker-dealers for the execution of transactions on behalf of the Client's account, provided that such broker-dealers are registered with the Securities and Exchange Commission (SEC) and are members of FINRA." 
"3. JKL Securities, LLC, shall serve as the introducing broker-dealer for the Client's account, and all transactions shall be cleared and settled through MNO Clearing Corporation."

</brokerage_example>

<example_format>
{
    "Custodian and Brokerage": {
        "Custodian": "(Information Pertinent to the Custodian agreements)",
        "Brokerage": "(Information Pertinent to the Brokerage agreements)"
    }
}
</example_format>

Also return your confidence level representing how confident you that you were able to accurately identify and extract the requested information. Your confidence value should be one of "High", "Medium", or "Low"
Use High Confidence in cases you have no doubt you have identified and captured ALL the correct information and there was no ambiguity and there is no need for a human to review
Use Medium Confidence in cases where you are reasonably confident you have identified most of the information, but there might be some ambiguity and a human review would be beneficial
Use Low Confidence in cases where you are not confident in the information you have captured, and a human needs to review


Think through each step of your thought process and write your thoughts down in <scratchpad> xml tags
return the valid json array with the extracted details in <output> xml tags, only including the valid json
return your confidence level (Low, Medium, or High) in <confidence> xml tags
record your explanation for your response and cite your work and reasoning in <show_work> xml tags, ask yourself did i capture ALL of the pertinent Custodian and Brokerage information in the agreement?

"""

    
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 3000,
        "temperature": 0.5,
        "system": system_prompt,
        "messages": [    
            {
                "role": "user",
                "content": f"<document> {content} </document>"
            }
        ]
    }

    prompt = json.dumps(prompt)

    print("------------------------------------------------------")

    response = bedrock.invoke_model(body=prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get('body').read())
    llmOutput=response_body['content'][0]['text']

    
    scratch = parse_xml(llmOutput, "scratchpad")
    output = parse_xml(llmOutput, "output")
    confidence = parse_xml(llmOutput, "confidence")
    show_work = parse_xml(llmOutput, "show_work")

    return scratch, output, confidence, show_work


def extract_fee_info(content):

    system_prompt="""
You are an AI Data Processor whose goal is to identify and extract key information from an Investment Management Agreement for a wealth management client. You will be provided a Wealth Management Agreement (DIMA, IMA, or MDIMA)
Using the document context of the provided agreement Extract the Fee Structure and Compensation details in the agreement and Categorize the Fee type
Categorize the Fee Structure into one of the provided categories (Dont make assumptions - if no objective is mentioned or there are reference to other documents not provided that would contain this information select "Other or Not Found" and use a "Low" confidence rating)
Return your Categorization in a valid json object using the <example_format> provided below 
If multiple categories apply, include all applicable categories in the json object

Only the below categories are valid options
<valid_categories>
Asset-based Fee
Tiered Asset-based Fee
Performance-based fee
Fixed Fee
Hourly or Project-based Fee
Subscription or Retainer Fee
Transaction-based Fee
Additional Expenses
Other or Not Found
</valid_categories>

<example_format>
{
  "Fee": {
    "Fee Structure": "(Identified Fee Structure, if multiple apply include as comma separated value)",
    "Compensation Details": "(Details of the Fee Structure and Compensation - Example: "The Investment Manager's fee shall be calculated based on the following tiered schedule: - First $1,000,000 of assets: [X%] - Next $2,000,000 of assets: [Y%] - Assets over $3,000,000: [Z%]")"
  }
}
</example_format>

Also return your confidence level representing how confident you that you were able to accurately identify and extract, and categorize the requested information. Your confidence value should be one of "High", "Medium", or "Low"
Use High Confidence in cases you have no doubt you have identified and captured ALL the correct information and there was no ambiguity and there is no need for a human to review
Use Medium Confidence in cases where you are reasonably confident you have identified most of the information, but there might be some ambiguity and a human review would be beneficial
Use Low Confidence in cases where you are not confident in the information you have captured or the data provided didn't include enough information, and a human needs to review


Think through each step of your thought process and write your thoughts down in <scratchpad> xml tags
return the valid json array with the extracted details in <output> xml tags, only including the valid json
return your confidence level (Low, Medium, or High) in <confidence> xml tags
record your explanation for your response and cite your work and reasoning in <show_work> xml tags

"""

    
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "temperature": 0.5,
        "system": system_prompt,
        "messages": [    
            {
                "role": "user",
                "content": f"<document> {content} </document>"
            }
        ]
    }

    prompt = json.dumps(prompt)

    print("------------------------------------------------------")

    response = bedrock.invoke_model(body=prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get('body').read())
    llmOutput=response_body['content'][0]['text']

    
    scratch = parse_xml(llmOutput, "scratchpad")
    output = parse_xml(llmOutput, "output")
    confidence = parse_xml(llmOutput, "confidence")
    show_work = parse_xml(llmOutput, "show_work")

    return scratch, output, confidence, show_work


def extract_effective_date(content):

    system_prompt="""
You are an AI Data Processor whose goal is to identify and extract key information from an Investment Management Agreement for a wealth management client. You will be provided a Wealth Management Agreement (DIMA, IMA, or MDIMA)
Using the document context of the provided agreement Identify and extract (making no changes)Effective Date of the Agreement
Also capture the signature dates for the Client and the Investment Firm
Return your Results in a valid json object using the <example_format> provided below

<example_format>
{
    "Effective Date": {
        "Effective Date": "(The Effective Date of the Agreement)",
        "Client Signature Date": "(The Date the Client Signed the Agreement)",
        "Investment Firm Signature Date": "(The Date the Investment Firm Signed the agreement)"
    }
}
</example_format>

Also return your confidence level representing how confident you that you were able to accurately identify and extract the requested information. Your confidence value should be one of "High", "Medium", or "Low"
Use High Confidence in cases you have no doubt you have identified and captured ALL the correct information and there was no ambiguity and there is no need for a human to review
Use Medium Confidence in cases where you are reasonably confident you have identified most of the information, but there might be some ambiguity and a human review would be beneficial
Use Low Confidence in cases where you are not confident in the information you have captured or the data provided didn't include enough information, and a human needs to review


Think through each step of your thought process and write your thoughts down in <scratchpad> xml tags
return the valid json array with the extracted details in <output> xml tags, only including the valid json
return your confidence level (Low, Medium, or High) in <confidence> xml tags
record your explanation for your response and cite your work and reasoning in <show_work> xml tags

"""

    
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 3000,
        "temperature": 0.5,
        "system": system_prompt,
        "messages": [    
            {
                "role": "user",
                "content": f"<document> {content} </document>"
            }
        ]
    }

    prompt = json.dumps(prompt)

    print("------------------------------------------------------")

    response = bedrock.invoke_model(body=prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get('body').read())
    llmOutput=response_body['content'][0]['text']

    
    scratch = parse_xml(llmOutput, "scratchpad")
    output = parse_xml(llmOutput, "output")
    confidence = parse_xml(llmOutput, "confidence")
    show_work = parse_xml(llmOutput, "show_work")

    return scratch, output, confidence, show_work


def final_json(party_json, objective_json, custodian_json, fee_json, effective_date_json, file_name):
    final_json = {
        "File Name": file_name,
        "Data": {
            "Party": party_json,
            "Objective": objective_json,
            "Custodian and Brokerage": custodian_json,
            "Fee": fee_json,
            "Effective Date": effective_date_json
        }
    }
    return final_json



#Setup Streamlit
st.set_page_config(page_title="MDIMA Extraction", page_icon=":tada", layout="wide")
st.title(f""":rainbow[Investment Management  Entity Extraction]""")

st.write("---")
uploaded_file = st.file_uploader('Upload a .pdf file', type="pdf")
st.write("---")

go=st.button("Go!")
if go and uploaded_file is not None:
    st.balloons()
    file_name = uploaded_file.name
    extract(uploaded_file, file_name)
