
import stun

nat_type, external_ip, external_port = stun.get_ip_info(stun_host="stun.l.google.com", stun_port=19302)
#nat_type, external_ip, external_port = stun.get_ip_info(stun_host="stun.minisipserver.com", stun_port=19302)
#nat_type, external_ip, external_port = stun.get_ip_info()
print(nat_type)
print(external_ip)
print(external_port)