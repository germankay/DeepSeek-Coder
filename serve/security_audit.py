import re

from pydantic import BaseModel, Field

CYBERCODE_SECURITY_SYSTEM_PROMPT = """\
Tu es un expert en cybersécurité chez CyberCode Studio. Analyse le code suivant et identifie les vulnérabilités potentielles. Pour chaque faille, indique la ligne, le type, la gravité (faible/moyenne/élevée/critique) et propose un correctif précis.
Code : {code}
Langage : {langage}
"""

class VulnerabilityItem(BaseModel):
    line: int = Field(default=0, description="Line number of the identified vulnerability.")
    vulnerability_type: str = Field(description="Type of vulnerability, e.g., SQL Injection, XSS, Hardcoded Secret.")
    severity: str = Field(description="Severity level: faible, moyenne, élevée, critique.")
    description: str = Field(description="Description of the security flaw.")
    recommendation: str = Field(description="Precise remediation steps or code fix.")

class SecurityAuditReport(BaseModel):
    language: str
    security_score: int = Field(ge=0, le=100, description="Overall security score out of 100.")
    vulnerabilities: list[VulnerabilityItem] = Field(default_factory=list)
    summary: str = Field(description="Executive summary of the security analysis.")

class SecurityAuditor:
    """
    Security Audit module combining rule-based heuristic patterns and LLM-ready prompt building.
    """
    def __init__(self):
        # Heuristic rules for common vulnerabilities
        self.rules = [
            {
                "pattern": r"(?i)(select|insert|update|delete)\s+.*\+\s*[\"\']?\w+",
                "type": "SQL Injection",
                "severity": "élevée",
                "desc": "String concatenation detected in SQL query construction.",
                "rec": "Use parameterized queries or prepared statements (ORMs)."
            },
            {
                "pattern": r"(?i)(eval|exec|system|popen|passthru|shell_exec)\s*\(",
                "type": "Code/Command Injection",
                "severity": "critique",
                "desc": "Execution of arbitrary code/command using eval/exec/system.",
                "rec": "Avoid dynamic code execution. Sanitize inputs and use safer APIs."
            },
            {
                "pattern": r"(?i)(api[_-]?key|secret|password|passwd|private_key)\s*[:=]\s*[\"\'][A-Za-z0-9_\-\.]{6,}[\"\']",
                "type": "Hardcoded Secret",
                "severity": "élevée",
                "desc": "Hardcoded credential or API secret found in source code.",
                "rec": "Store secrets in environment variables or a key management vault (HashiCorp Vault / AWS KMS)."
            },
            {
                "pattern": r"(?i)(document\.write|innerHTML|v-html)\s*=",
                "type": "Cross-Site Scripting (XSS)",
                "severity": "moyenne",
                "desc": "Unsanitized user content directly injected into DOM.",
                "rec": "Use safe DOM manipulation APIs or escape output properly."
            },
            {
                "pattern": r"(?i)md5\(|sha1\(",
                "type": "Weak Cryptographic Hash",
                "severity": "faible",
                "desc": "Use of weak cryptographic hashing function (MD5/SHA1).",
                "rec": "Upgrade to SHA-256, SHA-3, or Argon2/bcrypt for password hashing."
            }
        ]

    def build_audit_prompt(self, code: str, language: str = "auto") -> str:
        """
        Builds the exact system prompt instructed for CyberCode Studio audit.
        """
        return CYBERCODE_SECURITY_SYSTEM_PROMPT.format(code=code, langage=language)

    def analyze(self, code: str, language: str = "auto") -> SecurityAuditReport:
        """
        Performs static heuristic analysis on code and generates a structured report.
        """
        lines = code.split("\n")
        vulnerabilities: list[VulnerabilityItem] = []

        for line_num, line in enumerate(lines, start=1):
            for rule in self.rules:
                if re.search(rule["pattern"], line):
                    vulnerabilities.append(
                        VulnerabilityItem(
                            line=line_num,
                            vulnerability_type=rule["type"],
                            severity=rule["severity"],
                            description=rule["desc"],
                            recommendation=rule["rec"]
                        )
                    )

        # Calculate score: start at 100, deduct based on severity
        score = 100
        for v in vulnerabilities:
            if v.severity == "critique":
                score -= 30
            elif v.severity == "élevée":
                score -= 20
            elif v.severity == "moyenne":
                score -= 10
            elif v.severity == "faible":
                score -= 5

        score = max(0, score)

        if not vulnerabilities:
            summary = "Aucune vulnérabilité majeure n'a été détectée par l'analyse statique. Le code respecte les bonnes pratiques de base."
        else:
            summary = f"Analyse terminée : {len(vulnerabilities)} vulnérabilité(s) identifiée(s). Score global de sécurité : {score}/100."

        return SecurityAuditReport(
            language=language,
            security_score=score,
            vulnerabilities=vulnerabilities,
            summary=summary
        )

def audit_code_security(code: str, language: str = "auto") -> SecurityAuditReport:
    auditor = SecurityAuditor()
    return auditor.analyze(code, language)
