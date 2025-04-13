import os
import time
import argparse
import json
import requests
import random
from datetime import datetime
import urllib.parse
from tqdm import tqdm
import logging
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler("pixiv_crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PixivCrawler")

class PixivCrawlerNoLogin:
    def __init__(self, headless=True, download_delay=(1, 3), verbose=False):
        self.download_delay = download_delay
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info("Chrome 드라이버 초기화 중...")
        
        # Chrome 옵션 설정
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")  # 새 헤드리스 모드
        
        # 기본 옵션 설정
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")
        
        # 봇 탐지 우회를 위한 설정
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        # 랜덤 사용자 에이전트
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36"
        ]
        chrome_options.add_argument(f"--user-agent={random.choice(user_agents)}")
        
        try:
            # 최신 ChromeDriver 설치 및 사용
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # CDP 명령어로 자동화 표시 제거 추가
            self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
                """
            })
            
            self.driver.set_page_load_timeout(30)
            self.wait = WebDriverWait(self.driver, 15)  # 타임아웃 증가
            
            logger.info("Chrome 드라이버 초기화 완료")
        except Exception as e:
            logger.error(f"드라이버 초기화 오류: {e}")
            raise
        
        # 다운로드한 이미지 ID 저장용
        self.downloaded_ids = set()
            
    def __del__(self):
        try:
            if hasattr(self, 'driver'):
                self.driver.quit()
                logger.info("Chrome 드라이버 종료")
        except:
            pass
        
    def handle_cookies_and_popups(self):
        try:
            # 쿠키 동의 버튼이 있으면 클릭
            try:
                cookie_buttons = self.driver.find_elements(By.XPATH, 
                    "//*[contains(text(), 'Accept') or contains(text(), '동의') or contains(text(), 'Agree') or @data-click-label='agree']")
                
                for button in cookie_buttons:
                    if button.is_displayed():
                        logger.debug("쿠키 동의 버튼 클릭")
                        button.click()
                        time.sleep(2)
                        break
            except Exception as e:
                logger.debug(f"쿠키 동의 처리 중 예외 (무시됨): {e}")
            
            # 로그인 권장 팝업이 있으면 닫기
            try:
                close_buttons = self.driver.find_elements(By.XPATH, 
                    "//button[contains(@class, 'close') or contains(@class, 'dismiss') or contains(@class, 'cancel')]")
                
                for button in close_buttons:
                    if button.is_displayed():
                        logger.debug("팝업 닫기 버튼 클릭")
                        button.click()
                        time.sleep(1)
            except Exception as e:
                logger.debug(f"팝업 닫기 중 예외 (무시됨): {e}")
                
        except Exception as e:
            logger.warning(f"쿠키/팝업 처리 중 오류: {e}")
    
    def search_illustrations(self, query, sort="popular_desc", max_retries=3):
        # 정렬 매개변수 맵핑
        sort_param = {
            "popular_desc": "popular_d",
            "date_desc": "date_d",
            "date_asc": "date"
        }.get(sort, "popular_d")
        
        # 검색 URL 구성
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://www.pixiv.net/tags/{encoded_query}/illustrations?order={sort_param}&mode=safe"
        
        for attempt in range(max_retries):
            try:
                logger.info(f"검색 페이지로 이동 중: {search_url} (시도 {attempt+1}/{max_retries})")
                self.driver.get(search_url)
                
                # 쿠키 및 팝업 처리
                self.handle_cookies_and_popups()
                
                # 검색 결과 로드 대기
                for selector in ['div[role="presentation"] a > div', 'section ul li a', '.gtm-illust-recommend-thumbnail']:
                    try:
                        self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                        logger.info(f"검색 결과 로드 완료 (확인된 요소: {selector})")
                        
                        # 디버그용 스크린샷
                        if self.verbose:
                            self.driver.save_screenshot("search_results.png")
                            logger.debug("검색 결과 스크린샷 저장: search_results.png")
                        
                        return True
                    except TimeoutException:
                        continue
                
                # 모든 선택자 시도 후에도 실패
                if attempt < max_retries - 1:
                    logger.warning("검색 결과를 찾을 수 없습니다. 재시도 중...")
                    time.sleep(3)
                else:
                    logger.error("검색 결과를 찾을 수 없습니다.")
                    # 디버그용 스크린샷
                    self.driver.save_screenshot("search_error.png")
                    logger.debug("검색 오류 스크린샷 저장: search_error.png")
                    return False
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"검색 중 오류: {str(e)}. 재시도 중...")
                    time.sleep(3)
                else:
                    logger.error(f"검색 최종 실패: {str(e)}")
                    return False
        
        # 모든 시도 실패
        return False
    
    def extract_illustration_ids(self, max_items=100, max_scroll_attempts=30):
        illustration_ids = []
        
        # 스크롤하며 이미지 로드
        logger.info("일러스트 ID 수집 중...")
        last_height = 0
        consecutive_same_height = 0
        scroll_attempts = 0
        
        # 다양한 CSS 선택자 시도
        selectors = [
            'a[href^="/artworks/"]',
            'a[href*="/artworks/"]',
            'div[type="illust"] a',
            'li a[data-gtm-value]'
        ]
        
        while len(illustration_ids) < max_items and scroll_attempts < max_scroll_attempts:
            # 현재 로드된 일러스트 링크 추출
            all_links = []
            
            for selector in selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    all_links.extend(elements)
                except:
                    continue
            
            new_ids_found = False
            
            for element in all_links:
                try:
                    href = element.get_attribute('href')
                    if not href:
                        continue
                        
                    # /artworks/12345678 형식에서 ID만 추출
                    if '/artworks/' in href:
                        illust_id = href.split('/artworks/')[-1].split('/')[0].split('?')[0]
                        
                        if illust_id.isdigit() and illust_id not in illustration_ids:
                            illustration_ids.append(illust_id)
                            new_ids_found = True
                            
                            if len(illustration_ids) >= max_items:
                                break
                except Exception as e:
                    logger.debug(f"ID 추출 중 오류 (무시됨): {e}")
                    continue
            
            # 최대 항목 수에 도달하면 중단
            if len(illustration_ids) >= max_items:
                break
            
            # 새 ID를 찾지 못했고 연속으로 같은 높이가 3번 이상이면 추가 스크롤 없이 진행
            if not new_ids_found and consecutive_same_height >= 3:
                logger.info("더 이상 새로운 일러스트를 찾을 수 없습니다.")
                break
                
            # 스크롤 다운
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # 스크롤 후 콘텐츠 로드 대기
            scroll_attempts += 1
            
            # 무한 스크롤 종료 조건 확인
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                consecutive_same_height += 1
                if consecutive_same_height >= 3:  # 3번 연속으로 같은 높이면 종료
                    logger.info("더 이상 로드할 콘텐츠가 없습니다.")
                    break
            else:
                consecutive_same_height = 0
                
            last_height = new_height
            
            # 진행 상황 출력
            if len(illustration_ids) > 0 and (len(illustration_ids) % 10 == 0 or scroll_attempts % 5 == 0):
                logger.info(f"현재 {len(illustration_ids)}개의 일러스트 ID 수집됨... (스크롤 {scroll_attempts}/{max_scroll_attempts})")
        
        logger.info(f"총 {len(illustration_ids)}개의 일러스트 ID 수집 완료")
        return illustration_ids
    
    def get_illustration_details(self, illust_id, max_retries=3):
        artwork_url = f"https://www.pixiv.net/artworks/{illust_id}"
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"ID {illust_id}: 상세 페이지 로드 중... (시도 {attempt+1}/{max_retries})")
                self.driver.get(artwork_url)
                
                # 쿠키 및 팝업 처리
                self.handle_cookies_and_popups()
                
                # 작품 로드 대기
                figure_selectors = ['main figure', 'figure[role="presentation"]', 'div[role="presentation"] img']
                
                figure = None
                for selector in figure_selectors:
                    try:
                        figure = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                        break
                    except:
                        continue
                
                if not figure:
                    # 로그인 요청 페이지인지 확인
                    if "로그인" in self.driver.page_source or "login" in self.driver.page_source.lower():
                        logger.warning(f"ID {illust_id}: 로그인이 필요한 콘텐츠입니다.")
                        return None
                    
                    raise Exception("작품 이미지를 찾을 수 없습니다.")
                
                # 수집할 메타데이터 초기화
                title = ""
                artist = ""
                artist_id = ""
                tags = []
                
                # 제목 가져오기
                title_selectors = ['main h1', 'h1', 'figcaption h1', 'h1[title]']
                for selector in title_selectors:
                    try:
                        title_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        title = title_element.text or title_element.get_attribute('title')
                        if title:
                            break
                    except:
                        continue
                
                # 작가 정보 가져오기
                artist_selectors = ['main a[href^="/users/"]', 'aside a[href^="/users/"]']
                for selector in artist_selectors:
                    try:
                        artist_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        artist = artist_element.text
                        artist_href = artist_element.get_attribute('href')
                        if artist_href and '/users/' in artist_href:
                            artist_id = artist_href.split('/users/')[-1].split('/')[0]
                        if artist:
                            break
                    except:
                        continue
                
                # 태그 수집
                tag_selectors = ['footer a[href^="/tags/"]', 'aside a[href^="/tags/"]', 'nav a[href^="/tags/"]']
                for selector in tag_selectors:
                    try:
                        tag_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for tag_elem in tag_elements:
                            tag_text = tag_elem.text.strip()
                            if tag_text and tag_text not in tags:
                                tags.append(tag_text)
                        if tags:
                            break
                    except:
                        continue
                
                # 원본 이미지 URL 추출 시도
                
                # 방법 1: 작품 클릭하여 큰 이미지 보기
                try:
                    figure.click()
                    time.sleep(2)
                    
                    # 원본 이미지 로드 대기
                    img_selectors = ['div[role="presentation"] img', 'div[role="dialog"] img', 'img[alt]']
                    
                    img_element = None
                    for selector in img_selectors:
                        try:
                            img_element = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                            break
                        except:
                            continue
                    
                    if not img_element:
                        raise Exception("확대된 이미지를 찾을 수 없습니다.")
                    
                    img_url = img_element.get_attribute('src')
                    
                    # 원본 이미지 URL로 변환
                    if 'pximg.net' in img_url:
                        # small, medium, large, master를 original로 변경
                        replacements = [
                            ('/c/600x1200_90_webp/img-master/', '/img-original/'),
                            ('/img-master/', '/img-original/'),
                            ('/c/600x1200_90/img-master/', '/img-original/'),
                            ('/custom-thumb/', '/img-original/')
                        ]
                        
                        for old, new in replacements:
                            if old in img_url:
                                img_url = img_url.replace(old, new)
                                break
                        
                        # 마스터 파일 확장자 제거
                        img_url = img_url.replace('_master1200.jpg', '.jpg')
                        img_url = img_url.replace('_custom1200.jpg', '.jpg')
                        img_url = img_url.replace('_square1200.jpg', '.jpg')
                        
                        # PNG 확장자 경로도 준비
                        img_url_png = img_url.replace('.jpg', '.png')
                    else:
                        logger.warning(f"ID {illust_id}: 지원되지 않는 이미지 URL 형식")
                        img_url_png = img_url.replace('.jpg', '.png')
                    
                    # 뒤로 가기 (ESC 키 누르기)
                    webdriver.ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"ID {illust_id}: 확대 이미지 처리 중 오류: {e}")
                    img_url = ""
                    img_url_png = ""
                
                # 방법 2: 이미지를 찾지 못했거나 오류 발생 시, 페이지 소스에서 URL 추출 시도
                if not img_url or 'pximg.net' not in img_url:
                    try:
                        page_source = self.driver.page_source
                        
                        # 원본 이미지 URL 패턴 찾기
                        import re
                        original_urls = re.findall(r'https://i\.pximg\.net/img-original/img/[^"\']+\.(jpg|png)', page_source)
                        
                        if original_urls:
                            img_url = original_urls[0][0]
                            if img_url.endswith('.jpg'):
                                img_url_png = img_url.replace('.jpg', '.png')
                            else:
                                img_url_png = img_url
                                img_url = img_url.replace('.png', '.jpg')
                    except Exception as e:
                        logger.warning(f"ID {illust_id}: 페이지 소스 분석 중 오류: {e}")
                
                # 메타데이터 없이 성공했는지 확인
                if not img_url or 'pximg.net' not in img_url:
                    if attempt < max_retries - 1:
                        logger.warning(f"ID {illust_id}: 이미지 URL을 찾을 수 없습니다. 재시도 중...")
                        time.sleep(2)
                        continue
                    else:
                        logger.error(f"ID {illust_id}: 이미지 URL을 찾을 수 없습니다.")
                        return None
                
                # 이미지 데이터 반환
                result = {
                    "id": illust_id,
                    "title": title,
                    "artist": artist,
                    "artist_id": artist_id,
                    "img_url": img_url,
                    "img_url_png": img_url_png,
                    "tags": tags,
                    "artwork_url": artwork_url,
                    "created_at": datetime.now().isoformat()
                }
                
                logger.debug(f"ID {illust_id}: 메타데이터 수집 완료")
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"ID {illust_id} 상세정보 가져오기 실패: {str(e)}. 재시도 중...")
                    time.sleep(2)
                else:
                    logger.error(f"ID {illust_id} 상세정보 가져오기 최종 실패: {str(e)}")
                    # 디버그용 스크린샷
                    if self.verbose:
                        self.driver.save_screenshot(f"error_{illust_id}.png")
                        logger.debug(f"오류 스크린샷 저장: error_{illust_id}.png")
                    return None
        
        # 모든 시도 실패
        return None
    
    def download_image(self, img_url, save_path, max_retries=3):
        headers = {
            'Referer': 'https://www.pixiv.net/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(img_url, headers=headers, stream=True, timeout=30)
                response.raise_for_status()
                
                # 저장 디렉토리 확인 및 생성
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404 and img_url.endswith('.jpg'):
                    # JPG 실패 시 PNG 시도
                    png_url = img_url.replace('.jpg', '.png')
                    save_path = save_path.replace('.jpg', '.png')
                    logger.debug(f"JPG 형식 실패, PNG 형식 시도 중: {png_url}")
                    return self.download_image(png_url, save_path, max_retries - attempt)
                elif attempt < max_retries - 1:
                    logger.warning(f"다운로드 실패 (HTTP {e.response.status_code}): 재시도 중...")
                    time.sleep(2)
                else:
                    logger.error(f"다운로드 최종 실패 (HTTP {e.response.status_code})")
                    return False
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"다운로드 실패: {str(e)}. 재시도 중...")
                    time.sleep(2)
                else:
                    logger.error(f"다운로드 최종 실패: {str(e)}")
                    return False
        
        # 모든 시도 실패
        return False
    
    def crawl_and_download(self, query, limit=1000, output_dir="pixiv_images", sort="popular_desc"):
        # 메인 출력 디렉토리 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"메인 디렉토리 생성: {output_dir}")
        
        # 이미지 및 메타데이터 하위 디렉토리 생성
        images_dir = os.path.join(output_dir, "images")
        metadata_dir = os.path.join(output_dir, "metadata")
        
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
            logger.info(f"이미지 디렉토리 생성: {images_dir}")
            
        if not os.path.exists(metadata_dir):
            os.makedirs(metadata_dir)
            logger.info(f"메타데이터 디렉토리 생성: {metadata_dir}")
            
        # 검색 실행
        search_success = self.search_illustrations(query, sort)
        if not search_success:
            logger.error("검색 실패. 프로그램을 종료합니다.")
            return 0
            
        # 일러스트 ID 추출 (배치 처리)
        downloaded_count = 0
        batch_size = min(100, limit)
        
        while downloaded_count < limit:
            remaining = limit - downloaded_count
            current_batch = min(batch_size, remaining)
            
            # 일러스트 ID 추출
            illustration_ids = self.extract_illustration_ids(max_items=current_batch)
            
            if not illustration_ids:
                logger.warning("더 이상 일러스트를 찾을 수 없습니다.")
                break
                
            logger.info(f"{len(illustration_ids)}개의 일러스트에 대한 다운로드 시작...")
            
            # 각 일러스트에 대해 상세 정보 가져오기 및 다운로드
            for illust_id in tqdm(illustration_ids, desc="다운로드 중"):
                # 이미 다운로드한 ID 건너뛰기
                if illust_id in self.downloaded_ids:
                    logger.debug(f"ID {illust_id}: 이미 다운로드됨, 건너뜁니다.")
                    continue
                
                # 이미 파일이 존재하는지 확인 (이미지 디렉토리 내)
                existing_images = [
                    f for f in os.listdir(images_dir) 
                    if f.startswith(f"{illust_id}_") and (f.endswith(".png") or f.endswith(".jpg"))
                ]
                
                if existing_images:
                    logger.debug(f"ID {illust_id}: 이미지 파일이 이미 존재합니다: {existing_images[0]}")
                    self.downloaded_ids.add(illust_id)
                    downloaded_count += 1
                    continue
                    
                # 상세 정보 가져오기
                illust_details = self.get_illustration_details(illust_id)
                
                if not illust_details:
                    logger.warning(f"ID {illust_id}: 상세 정보를 가져올 수 없습니다. 건너뜁니다.")
                    continue
                
                # PNG 파일 다운로드 시도
                png_url = illust_details["img_url_png"]
                png_filename = f"{illust_id}_{illust_details['artist_id'] or 'unknown'}.png"
                png_save_path = os.path.join(images_dir, png_filename)
                
                # 먼저 PNG로 시도
                logger.debug(f"ID {illust_id}: PNG 다운로드 시도")
                png_success = self.download_image(png_url, png_save_path)
                
                if png_success:
                    # 메타데이터 저장
                    metadata_filename = f"{illust_id}_{illust_details['artist_id'] or 'unknown'}.json"
                    metadata_path = os.path.join(metadata_dir, metadata_filename)
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(illust_details, f, ensure_ascii=False, indent=2)
                        
                    downloaded_count += 1
                    self.downloaded_ids.add(illust_id)
                    logger.info(f"ID {illust_id}: PNG 다운로드 성공 ({downloaded_count}/{limit})")
                else:
                    # PNG 실패 시 JPG로 시도
                    jpg_url = illust_details["img_url"]
                    jpg_filename = f"{illust_id}_{illust_details['artist_id'] or 'unknown'}.jpg"
                    jpg_save_path = os.path.join(images_dir, jpg_filename)
                    
                    logger.debug(f"ID {illust_id}: JPG 다운로드 시도")
                    jpg_success = self.download_image(jpg_url, jpg_save_path)
                    
                    if jpg_success:
                        # 메타데이터 저장
                        metadata_filename = f"{illust_id}_{illust_details['artist_id'] or 'unknown'}.json"
                        metadata_path = os.path.join(metadata_dir, metadata_filename)
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(illust_details, f, ensure_ascii=False, indent=2)
                            
                        downloaded_count += 1
                        self.downloaded_ids.add(illust_id)
                        logger.info(f"ID {illust_id}: JPG 다운로드 성공 ({downloaded_count}/{limit})")
                    else:
                        logger.warning(f"ID {illust_id}: 이미지 다운로드 실패")
                
                # 목표 개수에 도달하면 종료
                if downloaded_count >= limit:
                    break
                
                # 딜레이 추가 (서버 부하 방지)
                delay = random.uniform(self.download_delay[0], self.download_delay[1])
                time.sleep(delay)
            
            # 목표 개수에 도달하면 종료
            if downloaded_count >= limit:
                break
                
            # 더 많은 결과를 로드하기 위해 추가 스크롤
            try:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
            except:
                break
        
        logger.info(f"\n다운로드 완료: {downloaded_count}개의 일러스트를 다운로드했습니다.")
        logger.info(f"이미지 저장 위치: {images_dir}")
        logger.info(f"메타데이터 저장 위치: {metadata_dir}")
        return downloaded_count

def main():
    parser = argparse.ArgumentParser(description='Pixiv에서 일러스트를 크롤링하여 다운로드합니다. (로그인 없음)')
    parser.add_argument('query', help='검색할 태그나 키워드')
    parser.add_argument('--limit', type=int, default=1000, help='다운로드할 최대 이미지 수 (기본값: 1000)')
    parser.add_argument('--output_dir', default='pixiv_images', help='이미지 저장 디렉토리 (기본값: "pixiv_images")')
    parser.add_argument('--sort', default='popular_desc', 
                      choices=['popular_desc', 'date_desc', 'date_asc'],
                      help='정렬 방식 (기본값: "popular_desc")')
    parser.add_argument('--no-headless', action='store_true', help='브라우저 창 표시 (기본값: 숨김)')
    parser.add_argument('--min-delay', type=float, default=1.0, 
                      help='다운로드 사이의 최소 지연 시간(초) (기본값: 1.0)')
    parser.add_argument('--max-delay', type=float, default=3.0, 
                      help='다운로드 사이의 최대 지연 시간(초) (기본값: 3.0)')
    parser.add_argument('--verbose', action='store_true', help='상세 로깅 활성화')
    
    args = parser.parse_args()
    
    try:
        # Pixiv 크롤러 초기화
        crawler = PixivCrawlerNoLogin(
            headless=not args.no_headless,
            download_delay=(args.min_delay, args.max_delay),
            verbose=args.verbose
        )
        
        # 크롤링 및 다운로드 실행
        crawler.crawl_and_download(
            query=args.query,
            limit=args.limit,
            output_dir=args.output_dir,
            sort=args.sort
        )
        
    except KeyboardInterrupt:
        logger.info("\n사용자에 의해 프로그램이 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        return 1
    finally:
        # 드라이버가 종료되도록 보장
        if 'crawler' in locals():
            del crawler
    
    return 0

if __name__ == "__main__":
    exit(main())
