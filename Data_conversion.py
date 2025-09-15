import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

class ExcelToJSONConverter:
    """
    Convert Excel workbook sheets to JSON format for RAG vector database preparation
    """
    
    def __init__(self, excel_file_path: str, output_dir: str = "json_output"):
        """
        Initialize the converter
        
        Args:
            excel_file_path: Path to the Excel file
            output_dir: Directory to store JSON outputs
        """
        self.excel_file_path = excel_file_path
        self.output_dir = output_dir
        self.metadata = {
            "source_file": excel_file_path,
            "conversion_timestamp": datetime.now().isoformat(),
            "sheets_processed": []
        }
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
    
    def clean_data(self, value: Any) -> Any:
        """Clean and standardize data values"""
        if pd.isna(value):
            return None
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float)):
            return value
        return str(value)
    
    def convert_sheet_to_json(self, sheet_name: str, df: pd.DataFrame) -> List[Dict]:
        """
        Convert a single sheet to JSON format
        
        Args:
            sheet_name: Name of the sheet
            df: DataFrame containing sheet data
            
        Returns:
            List of JSON records
        """
        json_records = []
        
        # Clean column names
        df.columns = df.columns.astype(str).str.strip()
        
        for index, row in df.iterrows():
            # Create base record structure
            record = {
                "id": f"{sheet_name}_{index + 1}",
                "sheet_name": sheet_name,
                "row_number": index + 1,
                "data": {},
                "metadata": {
                    "source_sheet": sheet_name,
                    "total_columns": len(df.columns),
                    "non_null_fields": 0
                }
            }
            
            # Process each column
            non_null_count = 0
            for col in df.columns:
                cleaned_value = self.clean_data(row[col])
                record["data"][col] = cleaned_value
                if cleaned_value is not None and cleaned_value != "":
                    non_null_count += 1
            
            record["metadata"]["non_null_fields"] = non_null_count
            
            # Create a text representation for RAG (searchable content)
            text_content = self.create_text_content(record["data"], sheet_name)
            record["text_content"] = text_content
            
            json_records.append(record)
        
        return json_records
    
    def create_text_content(self, data: Dict, sheet_name: str) -> str:
        """
        Create searchable text content from record data
        
        Args:
            data: Record data dictionary
            sheet_name: Name of the sheet
            
        Returns:
            Formatted text content for RAG
        """
        text_parts = [f"Sheet: {sheet_name}"]
        
        # Add non-null fields as searchable text
        for key, value in data.items():
            if value is not None and str(value).strip() != "":
                # Format key-value pairs for better searchability
                formatted_key = key.replace("_", " ").replace("-", " ").title()
                text_parts.append(f"{formatted_key}: {value}")
        
        return " | ".join(text_parts)
    
    def process_workbook(self) -> Dict:
        """
        Process the entire Excel workbook
        
        Returns:
            Summary of processing results
        """
        try:
            # Read all sheets from Excel file
            all_sheets = pd.read_excel(self.excel_file_path, sheet_name=None)
            
            processing_summary = {
                "total_sheets": len(all_sheets),
                "sheets_processed": {},
                "total_records": 0,
                "output_files": []
            }
            
            # Process each sheet
            for sheet_name, df in all_sheets.items():
                print(f"Processing sheet: {sheet_name}")
                
                # Skip empty sheets
                if df.empty:
                    print(f"Skipping empty sheet: {sheet_name}")
                    continue
                
                # Convert sheet to JSON
                json_records = self.convert_sheet_to_json(sheet_name, df)
                
                # Save individual sheet JSON
                sheet_output_file = os.path.join(self.output_dir, f"{sheet_name}_records.json")
                with open(sheet_output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_records, f, indent=2, ensure_ascii=False, default=str)
                
                # Update summary
                processing_summary["sheets_processed"][sheet_name] = {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "records_created": len(json_records),
                    "output_file": sheet_output_file
                }
                processing_summary["total_records"] += len(json_records)
                processing_summary["output_files"].append(sheet_output_file)
                
                self.metadata["sheets_processed"].append({
                    "sheet_name": sheet_name,
                    "records_count": len(json_records),
                    "columns": list(df.columns)
                })
            
            # Create combined JSON file for all records
            self.create_combined_json_file()
            
            # Create JSONL file (common format for vector databases)
            self.create_jsonl_file()
            
            # Save processing metadata
            self.save_metadata()
            
            return processing_summary
            
        except Exception as e:
            print(f"Error processing workbook: {str(e)}")
            raise
    
    def create_combined_json_file(self):
        """Create a single JSON file with all records from all sheets"""
        combined_records = []
        
        # Read all individual sheet JSON files
        for file_path in Path(self.output_dir).glob("*_records.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                sheet_records = json.load(f)
                combined_records.extend(sheet_records)
        
        # Save combined file
        combined_file = os.path.join(self.output_dir, "all_records_combined.json")
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_records, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Combined JSON file created: {combined_file}")
        return combined_file
    
    def create_jsonl_file(self):
        """Create JSONL file (one JSON per line) - common format for vector databases"""
        combined_records = []
        
        # Read all individual sheet JSON files
        for file_path in Path(self.output_dir).glob("*_records.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                sheet_records = json.load(f)
                combined_records.extend(sheet_records)
        
        # Save as JSONL
        jsonl_file = os.path.join(self.output_dir, "all_records.jsonl")
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for record in combined_records:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
        
        print(f"JSONL file created: {jsonl_file}")
        return jsonl_file
    
    def save_metadata(self):
        """Save processing metadata"""
        metadata_file = os.path.join(self.output_dir, "processing_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Metadata saved: {metadata_file}")

def main():
    """Main function to run the conversion"""
    
    # Configuration
    excel_file_path = "EPCL VEHS Data (Mar23 - Mar24).xlsx"  # Replace with your Excel file path
    output_directory = "json_output"
    
    # Check if file exists
    if not os.path.exists(excel_file_path):
        print(f"Excel file not found: {excel_file_path}")
        print("Please update the excel_file_path variable with the correct path to your Excel file")
        return
    
    try:
        # Create converter instance
        converter = ExcelToJSONConverter(excel_file_path, output_directory)
        
        # Process the workbook
        print("Starting Excel to JSON conversion...")
        summary = converter.process_workbook()
        
        # Print summary
        print("\n" + "="*50)
        print("CONVERSION SUMMARY")
        print("="*50)
        print(f"Total sheets processed: {summary['total_sheets']}")
        print(f"Total records created: {summary['total_records']}")
        print(f"Output directory: {output_directory}")
        
        print("\nSheet-wise breakdown:")
        for sheet_name, details in summary['sheets_processed'].items():
            print(f"  {sheet_name}: {details['records_created']} records")
        
        print("\nOutput files created:")
        for file_path in summary['output_files']:
            print(f"  {file_path}")
        
        print(f"\nAdditional files:")
        print(f"  {output_directory}/all_records_combined.json")
        print(f"  {output_directory}/all_records.jsonl")
        print(f"  {output_directory}/processing_metadata.json")
        
        print("\n" + "="*50)
        print("Conversion completed successfully!")
        print("Files are ready for vector database ingestion.")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

# Example usage for vector database preparation
def prepare_for_rag_vector_db():
    """
    Example function showing how to prepare the JSON data for RAG vector database
    """
    print("\nRAG Vector Database Preparation Tips:")
    print("1. Use the 'text_content' field for creating embeddings")
    print("2. The 'all_records.jsonl' file is ready for most vector databases")
    print("3. Each record has a unique 'id' for referencing")
    print("4. Metadata is preserved for filtering and context")
    print("5. Consider chunking large text_content if needed")
    
    # Example of how you might load the data for vector database
    output_dir = "json_output"
    jsonl_file = os.path.join(output_dir, "all_records.jsonl")
    
    if os.path.exists(jsonl_file):
        print(f"\nExample: Loading {jsonl_file} for vector database:")
        
        sample_records = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 3:  # Show first 3 records as example
                    record = json.loads(line)
                    sample_records.append(record)
        
        for i, record in enumerate(sample_records, 1):
            print(f"\nSample Record {i}:")
            print(f"  ID: {record['id']}")
            print(f"  Sheet: {record['sheet_name']}")
            print(f"  Text Content Preview: {record['text_content'][:100]}...")

if __name__ == "__main__":
    main()
    prepare_for_rag_vector_db()