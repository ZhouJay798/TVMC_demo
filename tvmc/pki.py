"""Private X.509 CA for the local demo. Root provisioning is an external trust step."""
from datetime import datetime, timedelta, timezone
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey


def now(): return datetime.now(timezone.utc)
def pem(cert): return cert.public_bytes(serialization.Encoding.PEM).decode()
def name(subject): return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)])
def usage(signing=False, receiving=False, ca=False):
    return x509.KeyUsage(digital_signature=signing, content_commitment=False,
        key_encipherment=False, data_encipherment=False, key_agreement=receiving,
        key_cert_sign=ca, crl_sign=ca, encipher_only=False, decipher_only=False)


class DemoCA:
    """Enrollment authority: only local preapproved identities, no public CSR API.

    Real identity checking must happen externally. Private key never exported by
    experiment artifacts. Production CA must be separated from the distributor.
    """
    def __init__(self):
        self.key=Ed25519PrivateKey.generate();t=now()
        self.root=(x509.CertificateBuilder().subject_name(name('TVMC demo root'))
            .issuer_name(name('TVMC demo root')).public_key(self.key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(t-timedelta(minutes=1))
            .not_valid_after(t+timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True,path_length=0),critical=True)
            .add_extension(usage(ca=True),critical=True).sign(self.key,None))
        self.revoked=set();self.number=0

    def issue(self, subject, public_key, purpose, days=30):
        if purpose not in ('identity','receiving','issuer'):raise ValueError('certificate purpose')
        expected=X25519PublicKey if purpose=='receiving' else Ed25519PublicKey
        if not isinstance(public_key,expected):raise ValueError('certificate key algorithm')
        t=now()
        cert=(x509.CertificateBuilder().subject_name(name(subject)).issuer_name(self.root.subject)
            .public_key(public_key).serial_number(x509.random_serial_number())
            .not_valid_before(t-timedelta(minutes=1)).not_valid_after(t+timedelta(days=days))
            .add_extension(x509.BasicConstraints(ca=False,path_length=None),critical=True)
            .add_extension(usage(signing=purpose!='receiving',receiving=purpose=='receiving'),critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.UniformResourceIdentifier(
                'urn:tvmc:'+purpose+':'+subject)]),critical=False).sign(self.key,None))
        return pem(cert)

    def crl(self):
        t=now();builder=(x509.CertificateRevocationListBuilder().issuer_name(self.root.subject)
            .last_update(t-timedelta(seconds=1)).next_update(t+timedelta(hours=24))
            .add_extension(x509.CRLNumber(self.number),critical=False))
        for serial in sorted(self.revoked):
            builder=builder.add_revoked_certificate(x509.RevokedCertificateBuilder()
                .serial_number(serial).revocation_date(t-timedelta(seconds=1)).build())
        return pem(builder.sign(self.key,None))

    def revoke(self, certificate):
        self.revoked.add(x509.load_pem_x509_certificate(certificate.encode()).serial_number)
        self.number+=1
        return self.crl()


class Trust:
    def __init__(self, root_pem, crl_pem, issuer_certificate, min_crl_number=0):
        # root_pem MUST be provisioned out of band, not taken from a request.
        self.root=x509.load_pem_x509_certificate(root_pem.encode())
        self.crl_pem=crl_pem;self.issuer_certificate=issuer_certificate
        self.min_crl_number=min_crl_number

    def validate(self, certificate, subject, purpose):
        try:
            root=self.root;t=now()
            if not isinstance(root.public_key(),Ed25519PublicKey):raise ValueError('root algorithm')
            root.public_key().verify(root.signature,root.tbs_certificate_bytes)
            if (not root.extensions.get_extension_for_class(x509.BasicConstraints).value.ca or
                not root.extensions.get_extension_for_class(x509.KeyUsage).value.key_cert_sign or
                not root.extensions.get_extension_for_class(x509.KeyUsage).value.crl_sign or
                not root.not_valid_before_utc<=t<root.not_valid_after_utc):raise ValueError('invalid root')
            crl=x509.load_pem_x509_crl(self.crl_pem.encode())
            root.public_key().verify(crl.signature,crl.tbs_certlist_bytes)
            if (crl.issuer!=root.subject or not crl.last_update_utc<=t<crl.next_update_utc or
                crl.extensions.get_extension_for_class(x509.CRLNumber).value.crl_number<self.min_crl_number):
                raise ValueError('stale or wrong CRL')
            cert=x509.load_pem_x509_certificate(certificate.encode())
            root.public_key().verify(cert.signature,cert.tbs_certificate_bytes)
            if (cert.issuer!=root.subject or cert.subject!=name(subject) or
                not cert.not_valid_before_utc<=t<cert.not_valid_after_utc or
                cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca or
                crl.get_revoked_certificate_by_serial_number(cert.serial_number) is not None):
                raise ValueError('wrong identity, expired or revoked certificate')
            allowed={x509.BasicConstraints.oid,x509.KeyUsage.oid,x509.SubjectAlternativeName.oid}
            if any(e.critical and e.oid not in allowed for e in cert.extensions):raise ValueError('unknown critical extension')
            roles=cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            if roles.get_values_for_type(x509.UniformResourceIdentifier)!=['urn:tvmc:'+purpose+':'+subject]:
                raise ValueError('wrong certificate role')
            u=cert.extensions.get_extension_for_class(x509.KeyUsage).value;pk=cert.public_key()
            if purpose=='receiving':
                if not isinstance(pk,X25519PublicKey) or not u.key_agreement or u.digital_signature:raise ValueError('not receiving key')
            elif not isinstance(pk,Ed25519PublicKey) or not u.digital_signature or u.key_agreement:
                raise ValueError('not signing key')
            if u.key_cert_sign or u.crl_sign:raise ValueError('leaf cannot act as CA')
            return pk.public_bytes_raw()
        except Exception as exc:
            raise ValueError('certificate verification failed: '+str(exc)) from exc

    def update_crl(self, crl_pem):
        # Accept a valid CRL even when it revokes the issuer itself. Future use
        # must then fail; never silently retain an older non-revoking CRL.
        try:
            crl=x509.load_pem_x509_crl(crl_pem.encode());t=now()
            self.root.public_key().verify(crl.signature,crl.tbs_certlist_bytes)
            number=crl.extensions.get_extension_for_class(x509.CRLNumber).value.crl_number
            if crl.issuer!=self.root.subject or not crl.last_update_utc<=t<crl.next_update_utc or number<self.min_crl_number:
                raise ValueError('stale or wrong CRL')
        except Exception as exc:
            raise ValueError('invalid CRL update') from exc
        self.crl_pem=crl_pem;self.min_crl_number=number

    def export(self):
        return dict(root_certificate=pem(self.root),crl=self.crl_pem,issuer_certificate=self.issuer_certificate)


class Registry(dict):
    def __init__(self, trust):super().__init__();self.trust=trust


def member_key(registry, i, purpose='identity'):
    if not isinstance(registry,Registry):raise ValueError('CA-verified registry required')
    if i not in registry:raise ValueError('unknown member')
    reg=registry[i]
    if purpose not in ('identity','receiving') or purpose+'_certificate' not in reg:
        raise ValueError('missing member certificate')
    if reg['i']!=i:raise ValueError('member identity mismatch')
    pk=registry.trust.validate(reg[purpose+'_certificate'],'U'+str(i),purpose)
    if reg[purpose]!=pk.hex():raise ValueError('registered public key was substituted')
    return pk


def issuer_key(registry, claimed=None):
    if not isinstance(registry,Registry):raise ValueError('CA-verified registry required')
    pk=registry.trust.validate(registry.trust.issuer_certificate,'distributor','issuer')
    if claimed is not None and pk!=claimed:raise ValueError('issuer public key was substituted')
    return pk
