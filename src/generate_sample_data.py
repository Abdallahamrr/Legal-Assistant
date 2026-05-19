"""
Generate synthetic legal case documents for testing the RAG pipeline.
Run: python generate_sample_data.py
This creates sample .txt files (treated as plain documents) in data/
Now fully expanded to cover all 10 capstone case scenarios.
"""

import json
from pathlib import Path

SAMPLE_CASES = [
    {
        "filename": "force_majeure_cases.txt",
        "content": """LEGAL MEMORANDUM — FORCE MAJEURE IN SUPPLY CONTRACTS
Source: Commercial Law Review, Vol. 12 | Page: 1

Force majeure clauses excuse contractual performance when extraordinary events prevent a party from fulfilling obligations. 
Courts have interpreted these clauses narrowly, requiring that: (1) the event be unforeseeable; (2) the event be beyond the party's control; 
(3) the event directly caused the failure to perform.

COVID-19 AND FORCE MAJEURE
Source: Commercial Law Review, Vol. 12 | Page: 2

The COVID-19 pandemic has been extensively litigated as a force majeure event. In JN Contemporary Art LLC v. Phillips Auctioneers LLC 
(S.D.N.Y. 2020), the court held that COVID-19 governmental shutdown orders constituted a 'governmental action' and thus triggered the 
force majeure clause. The court reasoned that the clause listed 'acts of government' and pandemic-related executive orders qualified.

However, in Aukema v. Chesapeake Appalachia LLC, courts noted that general economic hardship or market downturns are typically 
insufficient to trigger force majeure. The party invoking force majeure must demonstrate that performance was objectively impossible, 
not merely more expensive or inconvenient.

FRUSTRATION OF PURPOSE DOCTRINE
Source: Commercial Law Review, Vol. 12 | Page: 3

When force majeure clauses are narrow, parties may invoke frustration of purpose. This doctrine applies when: 
(1) a supervening event occurs after contract formation; (2) the event was unforeseeable; (3) the event destroys the 
principal purpose of the contract; (4) the frustration is not the fault of either party.

In Lloyd v. Murphy (1944), the California Supreme Court held that frustration must be 'substantial' — not merely making 
the contract less profitable. The doctrine is narrowly applied and rarely succeeds when the contract includes a force majeure clause.

GOVERNMENTAL ACTION CLAUSE INTERPRETATION
Source: Commercial Law Review, Vol. 12 | Page: 4

Clauses listing 'governmental action' or 'acts of government' are interpreted to include: executive orders, administrative regulations, 
legislative acts, and emergency declarations. Courts in Gap Inc. v. Ponte Gadea New York LLC held that COVID-19 closure orders issued 
by state governors constituted 'governmental action' sufficient to trigger force majeure.

DAMAGES WHEN FORCE MAJEURE APPLIES
Source: Commercial Law Review, Vol. 12 | Page: 5

When force majeure is successfully invoked, the non-performing party is typically excused from liability for damages. 
The buyer (Alpha Corp) would not be entitled to cover damages, lost profits, or consequential damages during the force majeure period. 
However, if performance becomes permanently impossible, either party may terminate the contract without penalty.
"""
    },
    {
        "filename": "california_employment_law.txt",
        "content": """CALIFORNIA EMPLOYMENT LAW — NON-COMPETE AND IP ASSIGNMENTS
Source: California Employment Law Handbook | Page: 1

CALIFORNIA BUSINESS AND PROFESSIONS CODE SECTION 16600
California Business and Professions Code § 16600 provides: 'Except as provided in this chapter, every contract by which anyone 
is restrained from engaging in a lawful profession, trade, or business of any kind is to that extent void.'

This statute is among the strongest anti-non-compete laws in the United States. In Edwards v. Arthur Andersen LLP (2008), 
the California Supreme Court held that § 16600 bans non-compete agreements broadly, rejecting the 'narrow restraint' exception 
previously recognized by lower courts. Non-compete clauses are void and unenforceable in California regardless of consideration.

CHOICE OF LAW AND CALIFORNIA PUBLIC POLICY
Source: California Employment Law Handbook | Page: 2

Even when an employment agreement contains a choice-of-law clause selecting another state's law, California courts will 
apply California law to non-compete disputes involving California employees. In Application Group Inc. v. Hunter Group Inc., 
the court held that California has a 'strong public policy' against non-competes that overrides foreign choice-of-law provisions.

A Texas choice-of-law provision would not override California § 16600 protections for a California-based employee. 
The employee's right to work freely is a fundamental California public policy interest.

CALIFORNIA LABOR CODE §§ 2870-2872 — EMPLOYEE INVENTIONS
Source: California Employment Law Handbook | Page: 3

California Labor Code § 2870 limits employer IP assignment clauses. Employers CANNOT claim ownership of inventions that:
(1) Do not relate to the employer's business or anticipated R&D;
(2) Do not result from work performed for the employer;
(3) Are developed entirely on the employee's own time with own equipment and resources.

All four conditions must be met for the § 2870 carve-out to apply. If an employee develops an ML algorithm on a personal laptop 
during evenings, unrelated to employer's business, using no employer resources, the invention belongs to the employee.

IP ASSIGNMENT CLAUSE OVERBREADTH
Source: California Employment Law Handbook | Page: 4

An IP assignment clause that purports to assign 'all inventions conceived during employment' without the § 2870 carve-out 
is overbroad and void to that extent under California law. Employers must include statutory carve-out language. 
Courts will sever overbroad provisions rather than enforce them entirely.

INJUNCTIVE RELIEF FOR NON-COMPETE VIOLATIONS
Source: California Employment Law Handbook | Page: 5

Since non-competes are void in California, no injunctive relief is available to enforce them. 
AB 2276 (2023) further strengthened protections by imposing penalties on employers who attempt to enforce void non-competes.
"""
    },
    {
        "filename": "gdpr_data_protection.txt",
        "content": """GDPR COMPLIANCE GUIDE — DATA PROCESSORS AND SECURITY OBLIGATIONS
Source: EU Data Protection Handbook | Page: 1

GDPR ARTICLE 4 — DATA PROCESSOR DEFINITION
A 'data processor' means a natural or legal person, public authority, agency, or other body which processes personal data 
on behalf of the controller. A 'joint controller' exists when two or more controllers jointly determine the purposes and 
means of processing (Article 26).

A cloud vendor operating under a Data Processing Agreement (DPA) is typically a data processor, not a joint controller, 
unless it independently determines the purposes of processing. Under the standard DPA arrangement described, 
the cloud vendor would be classified as a data processor.

GDPR ARTICLE 32 — SECURITY OF PROCESSING
Source: EU Data Protection Handbook | Page: 2

Article 32 requires controllers and processors to implement 'appropriate technical and organisational measures' to ensure 
security appropriate to the risk, including:
(a) Pseudonymisation and encryption of personal data;
(b) Ongoing confidentiality, integrity, availability, and resilience of systems;
(c) Ability to restore data after incidents;
(d) Process for regularly testing security measures.

The term 'industry-standard security measures' in a DPA is ambiguous and may not satisfy Article 32's specificity requirements. 
Supervisory authorities (e.g., Ireland's DPC, UK's ICO) have ruled that vague security language does not meet Article 32 requirements. 
Encryption should be explicitly specified.

GDPR ARTICLE 83 — ADMINISTRATIVE FINES
Source: EU Data Protection Handbook | Page: 3

Article 83(4) imposes fines up to €10 million or 2% of global annual turnover (whichever is higher) for processor violations.
Article 83(5) imposes fines up to €20 million or 4% of global annual turnover for more serious infringements including 
violations of basic processing principles and data subject rights.

A breach exposing 500,000 records would likely be assessed under Article 83(5), with fines up to €20M or 4% of global turnover.

LIABILITY CAPS IN DPAs
Source: EU Data Protection Handbook | Page: 4

Contractual liability caps between controller and processor do not affect supervisory authority fines. 
However, they may limit the controller's ability to seek indemnification from the processor for damages paid to data subjects. 
Whether a liability cap applies depends on whether the processor's negligence contributed to the breach.

DATA BREACH NOTIFICATION
Source: EU Data Protection Handbook | Page: 5

Article 33 requires notification to supervisory authority within 72 hours of becoming aware of a personal data breach. 
Article 34 requires notification to data subjects when the breach is likely to result in high risk to their rights and freedoms. 
Failure to notify is a separate GDPR violation subject to additional fines.
"""
    },
    {
        "filename": "liquidated_damages_cases.txt",
        "content": """LIQUIDATED DAMAGES VS PENALTY CLAUSES — COMPARATIVE ANALYSIS
Source: Contract Law Quarterly | Page: 1

ENGLISH LAW — CAVENDISH SQUARE TEST
In Cavendish Square Holding BV v. Makdessi [2015] UKSC 67, the UK Supreme Court reformulated the penalty clause rule. 
The test is whether the clause imposes a detriment on the contract breaker 'out of all proportion' to any legitimate interest 
of the innocent party in the enforcement of the primary obligation. This replaced the older 'genuine pre-estimate of loss' test 
from Dunlop Pneumatic Tyre Co v. New Garage & Motor Co [1915].

AMERICAN LAW — UNIFORM COMMERCIAL CODE
Source: Contract Law Quarterly | Page: 2

Under UCC § 2-718 and common law, a liquidated damages clause is enforceable if:
(1) Actual damages were difficult to estimate at the time of contracting; AND
(2) The stipulated amount is a reasonable forecast of compensatory damages.

If either condition fails, the clause is an unenforceable penalty. Courts look to whether the clause represents a 
good-faith attempt to anticipate loss, not a punishment for breach.

$50/PACKAGE/DAY ANALYSIS
Source: Contract Law Quarterly | Page: 3

For the e-commerce hypothetical ($50/package/day during peak holiday season), enforceability depends on:
- Whether lost holiday sales were genuinely difficult to calculate at contracting (likely yes — dependent on consumer behavior)
- Whether $50/day represents a reasonable estimate of actual loss (lost sale, customer goodwill, reputational damage)

Courts have enforced per-unit, per-day delay damages in logistics contracts when the commercial context justifies them. 
See UPS Supply Chain Solutions v. Megatrux Transportation.

ACTUAL DAMAGES MITIGATION
Source: Contract Law Quarterly | Page: 4

Even if liquidated damages are upheld, the non-breaching party has a duty to mitigate. If the platform could have sourced 
alternative logistics for 8,000 packages and failed to do so, courts may reduce the award. 
The $2,000,000 total must be weighed against whether mitigation was attempted.
"""
    },
    {
        "filename": "trade_secrets_whistleblower.txt",
        "content": """TRADE SECRETS AND WHISTLEBLOWER PROTECTIONS
Source: Federal Employment & IP Law Review | Page: 1

DEFEND TRADE SECRETS ACT (DTSA) — 18 U.S.C. § 1836
The DTSA provides a federal civil cause of action for trade secret misappropriation. A trade secret includes:
(1) Information that derives economic value from not being generally known; AND
(2) Subject to reasonable measures to maintain secrecy.

Customer lists, pricing structures, and deal pipeline information have been held to qualify as trade secrets when:
- The company took reasonable steps to protect the information (access controls, NDAs, confidentiality training)
- The information was not readily ascertainable by competitors
See Fail-Safe LLC v. A.O. Smith Corp. (E.D. Wis. 2010).

MISAPPROPRIATION BY DEPARTING EMPLOYEES
Source: Federal Employment & IP Law Review | Page: 2

Emailing confidential data to a personal account before departing constitutes misappropriation under the DTSA. 
In PQ Labs Inc. v. Yang Qi, the court held that downloading trade secret data to personal devices days before resignation, 
then using that data at a competitor, constituted willful and malicious misappropriation justifying enhanced damages.

REMEDIES UNDER DTSA
Source: Federal Employment & IP Law Review | Page: 3

Available remedies include:
(1) Injunctive relief — prevent use or disclosure of the trade secret;
(2) Compensatory damages — actual loss or unjust enrichment;
(3) Exemplary damages — up to 2x compensatory damages for willful misappropriation;
(4) Attorney's fees — if misappropriation was willful and malicious.

WHISTLEBLOWER RETALIATION — SARBANES-OXLEY
Source: Federal Employment & IP Law Review | Page: 4

Section 806 of the Sarbanes-Oxley Act (SOX) protects employees of publicly traded companies who report violations of 
securities laws, SEC rules, or fraud. Key requirements:
(1) The employer must be a publicly traded company or a subsidiary/contractor thereof;
(2) The employee engaged in protected activity (reporting to supervisor, law enforcement, or Congress);
(3) The employer knew of the protected activity;
(4) The discharge or adverse action was caused by the protected activity.

DODD-FRANK WHISTLEBLOWER PROTECTIONS
Source: Federal Employment & IP Law Review | Page: 5

Dodd-Frank Section 922 extends protections to employees who report securities violations to the SEC. Importantly, 
the Supreme Court in Digital Realty Trust v. Somers (2018) held that Dodd-Frank protections apply ONLY to employees 
who report externally to the SEC, not to internal reports. For a private company, SOX protections are limited; 
Dodd-Frank does not apply unless the company is publicly traded or the employee reports to the SEC.
"""
    },
    {
        "filename": "saas_auto_renewal_disputes.txt",
        "content": """SAAS ENTERPRISE CONTRACTS — AUTO-RENEWAL AND CONSPICUOUS NOTICE
Source: Technology Transactions Review | Page: 1

AUTO-RENEWAL (EVERGREEN) CLAUSES IN B2B CONTRACTS
Auto-renewal clauses automatically extend the contract term unless a party gives notice of non-renewal. 
In enterprise software contracts, 90-day cancellation windows are common. Section 12.4 auto-renewals are generally 
enforceable in B2B transactions under general principles of contract law, as commercial parties are presumed to read and understand 
their agreements. Unlike consumer contracts, B2B agreements are not subject to standard state-level automatic renewal laws (ARLs) 
such as California's ARL (Bus. & Prof. Code § 17600), which requires conspicuous notice.

STATUTE OF FRAUDS AND ELECTRONIC SIGNATURES (E-SIGN ACT)
Source: Technology Transactions Review | Page: 2

Under the federal E-SIGN Act (15 U.S.C. § 7001) and state Uniform Electronic Transactions Acts (UETA), electronic signatures 
have the same legal status as handwritten signatures. An electronic signature satisfies the Statute of Frauds requirement 
for contracts that cannot be performed within one year. Once a 40-page contract is electronically signed, all terms, 
including buried Section 12.4 evergreen clauses, become binding.

CONSPICUOUS NOTICE EXCEPTIONS IN B2B DISPUTES
Source: Technology Transactions Review | Page: 3

Commercial parties occasionally attempt to invalidate buried clauses using the doctrine of unconscionability. However, 
courts rarely find unconscionability in B2B settings unless there is a severe disparity in bargaining power. 
In enterprise transactions, a $120,000 charge is enforceable if the party failed to track the non-renewal deadline. 
See Oracle Corp. v. Bexar County (W.D. Tex. 2012), enforcing auto-renewal terms signed electronically by commercial entities.
"""
    },
    {
        "filename": "whistleblower_wrongful_termination.txt",
        "content": """WHISTLEBLOWING & AT-WILL EMPLOYMENT EXCEPTIONS
Source: Employment Litigation Journal | Page: 1

AT-WILL EMPLOYMENT DOCTRINE & RETALIATION
At-will employment allows employers to terminate employees at any time for any legal reason. However, 
courts recognize a significant exception for public policy violations (the Tameny doctrine in California). 
An employer cannot terminate an employee in retaliation for: (1) refusing to violate a statute; (2) performing a statutory obligation; 
(3) reporting a statutory violation (whistleblowing) to a CEO or government agency.

FABRICATED PERFORMANCE IMPROVEMENT PLANS (PIPs)
Source: Employment Litigation Journal | Page: 2

In wrongful termination cases, employers often cite a Performance Improvement Plan (PIP) as a legitimate, non-retaliatory 
reason for discharge. To succeed, the employee must show that the PIP was a 'pretext' for discrimination or retaliation. 
Evidence of pretext includes: PIP initiated immediately after reporting a safety issue, arbitrary performance metrics, 
or positive reviews preceding the report. See Green v. Ralee Engineering Co. (1998) (safety concerns over aircraft defects).

SARBANES-OXLEY AND PRIVATE COMPANIES
Source: Employment Litigation Journal | Page: 3

While Sarbanes-Oxley (SOX) Section 806 primarily protects whistleblowers at publicly traded companies, the U.S. Supreme Court 
in Lawson v. FMR LLC (2014) held that SOX protections extend to employees of *private* contractors and subcontractors 
of public companies. If a private company is an active contractor for a public entity, SOX protections apply. 
However, for an independent private company, general state-level whistleblower statutes (e.g., California Labor Code § 1102.5) 
provide similar robust protection against termination for internal safety reports.
"""
    },
    {
        "filename": "construction_scope_creep.txt",
        "content": """CONSTRUCTION LAW — FIXED PRICE CONTRACTS AND CHANGE ORDERS
Source: Construction Jurisprudence | Page: 1

VALIDITY OF CHANGE ORDERS UNDER AIA DOCUMENT A201
Under standard American Institute of Architects (AIA) Document A201 General Conditions, a valid Change Order requires 
written agreement signed by the Owner, Contractor, and Architect. The document must describe the work added, 
the adjustment to the contract sum, and the adjustment to the contract time. 

VERBAL INSTRUCTIONS AND WAIVER CLAUSES
Source: Construction Jurisprudence | Page: 2

Most fixed-price construction contracts contain a clause stating that no changes are valid unless approved in writing. 
However, courts may find that the owner 'waived' the written change order requirement by: 
(1) verbally ordering the changes; (2) observing the contractor perform the extra work without objection; 
(3) verbally promising to pay for the changes. Under the doctrine of promissory estoppel or course of performance, 
a contractor can recover $1.2M for verbally ordered work if they can prove owner authorization.

UNJUST ENRICHMENT (QUANTUM MERUIT)
Source: Construction Jurisprudence | Page: 3

If a contract's change order clauses are strictly enforced against the contractor, the contractor may seek recovery under 
quantum meruit (unjust enrichment). This quasi-contractual remedy allows recovery of the reasonable value of services rendered 
to prevent the owner from receiving a windfall. The contractor must prove that the owner requested and accepted the benefits 
of the 23 change orders, and that it would be inequitable for the owner to retain them without payment.
"""
    },
    {
        "filename": "trade_secrets_misappropriation.txt",
        "content": """TRADE SECRET MISAPPROPRIATION & DTSA INJUNCTIVE RELIEF
Source: Intellectual Property Law Reporter | Page: 1

TRADE SECRET STATUS OF CUSTOMER LISTS
Under the Defend Trade Secrets Act (DTSA) (18 U.S.C. § 1836), customer lists, pricing databases, and deal pipelines 
are eligible for trade secret status. The plaintiff must show: (1) the list contains information not generally known; 
(2) the list has independent economic value; (3) the company used reasonable efforts to maintain secrecy. 
Reasonable efforts include: password protection, encrypted cloud drives, strictly enforced NDAs, and disabling USB ports. 
See Morlife, Inc. v. Perry (1997) (customer lists compiled through substantial time and expense qualify as trade secrets).

MISAPPROPRIATION BY EMAIL TO PERSONAL ACCOUNT
Source: Intellectual Property Law Reporter | Page: 2

An executive emailing 5,000 contacts and pricing structures to a personal email address immediately prior to resignation 
constitutes 'acquisition by improper means' and 'use without consent' under the DTSA. 
In SunPower Corp. v. SolarCity Corp. (2012), the court held that copying commercial data to personal accounts 
triggers clear misappropriation liability.

DTSA INJUNCTIVE RELIEF AND ATTORNEY FEES
Source: Intellectual Property Law Reporter | Page: 3

Under 18 U.S.C. § 1836(b)(3), courts may grant an injunction to prevent actual or threatened misappropriation, 
provided the injunction does not restrain a person from entering a lawful employment (complying with California public policy). 
If the misappropriation is proved to be willful and malicious, the court may award: 
(1) double exemplary damages; (2) reasonable attorney's fees to the prevailing party.
"""
    },
    {
        "filename": "arbitration_class_action_waiver.txt",
        "content": """ARBITRATION CLAUSES & CLASS ACTION WAIVERS
Source: Consumer Rights & Commercial Arbitration | Page: 1

UNCONSCIONABILITY OF ONLINE MANDATORY ARBITRATION
Online terms of service commonly require consumers to submit disputes to binding individual arbitration. 
A clause is 'unconscionable' if it is both procedurally unconscionable (take-it-or-leave-it adhesion contract) 
and substantively unconscionable (overly one-sided terms). When individual consumer recovery is small (e.g., $15/month) 
and arbitration filing fees exceed the recovery value, consumers argue the clause is unconscionable as it effectively 
forecloses any remedy.

AT&T MOBILITY V. CONCEPCION (2011) AND THE FAA
Source: Consumer Rights & Commercial Arbitration | Page: 2

In AT&T Mobility LLC v. Concepcion (2011), the U.S. Supreme Court held that the Federal Arbitration Act (FAA) 
preempts state laws that declare class-action waivers in arbitration agreements unconscionable. The Court held that 
arbitration agreements must be enforced according to their terms, even if they foreclose class actions and make small-value claims 
economically impractical to pursue individually.

PUBLIC POLICY EXCEPTIONS AND STATUTORY CLAIMS
Source: Consumer Rights & Commercial Arbitration | Page: 3

Despite *Concepcion*, narrow challenges remain. Arbitration clauses may be invalidated if they contain terms that prevent 
the vindication of statutory rights (the 'effective vindication' doctrine). However, the Supreme Court in American Express Co. 
v. Italian Colors Restaurant (2013) held that the effective vindication exception does not apply simply because it is 
economically irrational to arbitrate a small-value claim. Mandatory arbitration clauses with class action waivers remain 
extremely robust in commercial and consumer contracts.
"""
    }
]

def generate_sample_data(output_dir: str = "data/"):
    Path(output_dir).mkdir(exist_ok=True)
    for case in SAMPLE_CASES:
        path = Path(output_dir) / case["filename"]
        path.write_text(case["content"], encoding="utf-8")
        print(f"[Data] Created {path}")
    print(f"[Data] Generated {len(SAMPLE_CASES)} sample legal documents in {output_dir}/")

if __name__ == "__main__":
    generate_sample_data()
