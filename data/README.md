# ISM Source Documents

This folder contains the Australian Government Information Security Manual (ISM), December 2025 edition.

These are the raw source PDFs before any parsing, chunking, or embedding. The ingestion pipeline reads from this folder to populate the Supabase vector database.

## Document Set

Files 01 through 25 are used. File 00 (the consolidated document) is not included because the individual guideline documents are more suitable for chunking and retrieval and avoid duplicate content.

| File | Document |
|------|----------|
| 01 | Using the Information security manual |
| 02 | Cyber security principles |
| 03 | Guidelines for cyber security roles |
| 04 | Guidelines for cyber security incidents |
| 05 | Guidelines for procurement and outsourcing |
| 06 | Guidelines for cyber security documentation |
| 07 | Guidelines for physical security |
| 08 | Guidelines for personnel security |
| 09 | Guidelines for communications infrastructure |
| 10 | Guidelines for communications systems |
| 11 | Guidelines for enterprise mobility |
| 12 | Guidelines for evaluated products |
| 13 | Guidelines for information technology equipment |
| 14 | Guidelines for media |
| 15 | Guidelines for system hardening |
| 16 | Guidelines for system management |
| 17 | Guidelines for system monitoring |
| 18 | Guidelines for software development |
| 19 | Guidelines for database systems |
| 20 | Guidelines for email |
| 21 | Guidelines for networking |
| 22 | Guidelines for cryptography |
| 23 | Guidelines for gateways |
| 24 | Guidelines for data transfers |
| 25 | Cyber security terminology |

## Source

Published by the Australian Signals Directorate (ASD). Publicly available at https://www.cyber.gov.au/business-government/asds-cyber-security-frameworks/ism

## ISM Control Format

Each control in the ISM follows this format:

```
Control: ISM-XXXX; Revision: N; Updated: Mon-YY; Applicable: [NC/OS/P/S/TS]; Essential 8: [ML1/ML2/ML3/N/A]
```

Applicability levels:
- NC: Non-Classified
- OS: OFFICIAL:Sensitive
- P: PROTECTED
- S: SECRET
- TS: TOP SECRET
