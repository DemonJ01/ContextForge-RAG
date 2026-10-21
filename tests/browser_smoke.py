from playwright.sync_api import sync_playwright, expect
from pathlib import Path
from datetime import datetime, timezone
import json, os, tempfile

BASE=os.getenv('RAG_BASE_URL','http://127.0.0.1:8000')
OUT=Path(os.getenv('RAG_BROWSER_OUTPUT', str(Path(tempfile.gettempdir()) / 'rag-browser-results')))
OUT.mkdir(exist_ok=True)
checks=[]
errors=[]
failed_requests=[]

def record(name, details=True): checks.append({'check':name,'result':details})

def no_overflow(page, label):
    result=page.evaluate('''() => ({width:innerWidth,scrollWidth:document.documentElement.scrollWidth})''')
    assert result['scrollWidth'] <= result['width'], (label,result)
    return result

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    context=browser.new_context(viewport={'width':1440,'height':960}, permissions=['clipboard-read','clipboard-write'])
    page=context.new_page()
    page.on('pageerror', lambda error:errors.append(str(error)))
    page.on('requestfailed',lambda request:failed_requests.append({'url':request.url,'failure':request.failure}))
    page.goto(BASE,wait_until='networkidle')
    expect(page.locator('#connection-label')).to_have_text('Local · extractive')
    expect(page.locator('#library-count')).to_have_text('11')
    expect(page.locator('#send-button')).to_be_disabled()
    no_overflow(page,'desktop_empty')
    page.screenshot(path=str(OUT/'rag-desktop.png'),full_page=False,animations='disabled')
    record('live_connection_and_real_index_counts')
    # Shift+Enter leaves a draft, Enter submits once to the real API.
    question=page.locator('#question')
    question.fill('How do Kubernetes CronJobs work?')
    question.press('Shift+Enter')
    assert '\n' in question.input_value()
    with page.expect_response(lambda response: response.url.endswith('/query') and response.request.method=='POST') as live:
        question.press('Enter')
    assert live.value.status==200
    answer=live.value.json()
    assert answer['mode']=='local' and answer['status']=='complete' and answer['citations']
    expect(page.locator('.message-answer .answer-content')).to_have_count(1)
    expect(page.locator('#evidence-panel')).to_be_visible()
    expect(page.locator('#evidence-panel .evidence-card')).to_have_count(len(answer['citations']))
    assert 'Cron' in page.locator('.answer-content').inner_text()
    no_overflow(page,'desktop_answer')
    page.screenshot(path=str(OUT/'rag-answer.png'),full_page=False,animations='disabled')
    record('question_to_real_api_answer_and_evidence',{'citations':len(answer['citations']),'mode':answer['mode']})
    # Open exact document text from the citation rail.
    page.locator('.evidence-open').first.click()
    expect(page.locator('#document-dialog')).to_be_visible()
    expect(page.locator('#document-content')).to_contain_text(answer['citations'][0]['excerpt'])
    page.keyboard.press('Escape')
    expect(page.locator('#document-dialog')).not_to_be_visible()
    record('citation_opens_resolving_source_document')
    page.get_by_role('button',name='Copy answer',exact=True).click()
    assert page.evaluate('navigator.clipboard.readText()')==answer['answer']
    record('copy_answer')
    # Browse/filter/search the actual document metadata.
    page.locator('#library-nav').click()
    expect(page.locator('#document-list .document-row')).to_have_count(11)
    page.locator('#document-filter').select_option('reference')
    expect(page.locator('#document-list .document-row')).to_have_count(6)
    page.locator('#document-search').fill('Databricks')
    expect(page.locator('#document-list .document-row')).to_have_count(1)
    page.locator('#document-list .document-row').click()
    expect(page.locator('#document-content')).to_contain_text('Databricks CLI')
    page.keyboard.press('Escape')
    page.locator('#document-search').fill('no-such-document-xyz')
    expect(page.get_by_role('button',name='Reset filters',exact=True)).to_be_visible()
    page.get_by_role('button',name='Reset filters',exact=True).click()
    expect(page.locator('#document-list .document-row')).to_have_count(11)
    record('library_filter_search_details_and_empty_recovery')
    # New conversation and persisted browser-local history.
    page.locator('#new-chat').click()
    expect(page.locator('#empty-state')).to_be_visible()
    expect(page.locator('#messages')).not_to_be_visible()
    expect(page.locator('#conversation-list .history-item')).to_have_count(1)
    page.locator('#conversation-list .history-item').first.click()
    expect(page.locator('.message-answer .answer-content')).to_have_count(1)
    page.reload(wait_until='networkidle')
    expect(page.locator('.message-answer .answer-content')).to_have_count(1)
    record('new_conversation_history_and_reload_restore')
    # An unsupported question must honestly return no evidence.
    page.locator('#new-chat').click()
    page.locator('#question').fill('Explain quantum entanglement')
    with page.expect_response(lambda response: response.url.endswith('/query')) as unmatched:
        page.locator('#send-button').click()
    no_answer=unmatched.value.json()
    assert no_answer['status']=='no_answer' and no_answer['citations']==[]
    expect(page.locator('.answer-content')).to_contain_text("couldn't find enough evidence")
    assert page.locator('.answer-actions').get_by_role('button',name='sources').count()==0
    record('unsupported_question_no_evidence')
    # Simulated HTTP failure is an error UI; retry then reaches the real backend.
    page.locator('#new-chat').click()
    def fail_query(route): route.fulfill(status=503,content_type='application/json',body=json.dumps({'detail':'Temporary verification failure. Please retry.'}))
    page.route('**/query',fail_query)
    page.locator('#question').fill('How do I monitor Kubernetes job logs?')
    page.locator('#send-button').click()
    expect(page.locator('.error-message')).to_be_visible()
    expect(page.locator('.error-message')).to_contain_text('Temporary verification failure')
    page.unroute('**/query',fail_query)
    with page.expect_response(lambda response: response.url.endswith('/query')) as retried:
        page.locator('.error-message').get_by_role('button',name='Try again',exact=True).click()
    assert retried.value.status==200
    expect(page.locator('.error-message')).to_have_count(0)
    expect(page.locator('.message-answer .answer-content')).to_have_count(1)
    record('http_error_and_successful_real_retry')
    # Check modal keyboard focus, touch layout, and widths with persisted answer state.
    results=[]
    for width,height in [(1440,960),(1024,900),(768,1024),(375,812),(320,740)]:
        page.set_viewport_size({'width':width,'height':height})
        if page.locator('#evidence-panel').is_visible(): page.locator('#close-evidence').click()
        results.append(no_overflow(page,str(width)))
        if width<=768:
            page.locator('#mobile-menu').click()
            expect(page.locator('#sidebar')).to_have_attribute('aria-modal','true')
            page.keyboard.press('Escape')
            expect(page.locator('#mobile-menu')).to_have_attribute('aria-expanded','false')
        # Sources open as a focus-managed sheet on small screens.
        page.locator('.answer-actions').get_by_role('button',name='sources').first.click()
        expect(page.locator('#evidence-panel')).to_be_visible()
        if width<=1100:
            expect(page.locator('#evidence-panel')).to_have_attribute('aria-modal','true')
            assert page.evaluate("document.getElementById('evidence-panel').contains(document.activeElement)")
            page.keyboard.press('Escape')
            expect(page.locator('#evidence-panel')).not_to_be_visible()
        else: page.locator('#close-evidence').click()
        if width==375: page.screenshot(path=str(OUT/'rag-mobile.png'),full_page=False,animations='disabled')
    record('responsive_widths_and_keyboard_drawers',results)
    page.emulate_media(reduced_motion='reduce')
    assert page.evaluate("matchMedia('(prefers-reduced-motion: reduce)').matches")
    record('reduced_motion_preference')
    # Test untrusted source strings as inert text, not HTML.
    page.set_viewport_size({'width':1440,'height':960})
    page.locator('#new-chat').click()
    def hostile_answer(route):
        route.fulfill(status=200,content_type='application/json',body=json.dumps({'answer':'<img src=x onerror="window.ragXss=1"> [1]','mode':'local','elapsed_ms':1,'status':'complete','sources':['<script>window.ragXss=2</script>'],'citations':[{'id':'test','document_id':'nonexistent','title':'<img src=x onerror="window.ragXss=3">','excerpt':'<script>window.ragXss=4</script>'}]}))
    page.route('**/query',hostile_answer)
    page.locator('#question').fill('Source rendering verification')
    page.locator('#send-button').click()
    expect(page.locator('.answer-content')).to_contain_text('<img src=x')
    assert page.evaluate('window.ragXss') is None
    assert page.locator('.answer-content img').count()==0
    page.unroute('**/query',hostile_answer)
    record('untrusted_answer_and_source_html_inert')
    # A result received while browsing the library must appear on return.
    page.locator('#new-chat').click()
    held_queries=[]
    def hold_query(route): held_queries.append(route)
    page.route('**/query',hold_query)
    page.locator('#question').fill('How do Kubernetes CronJobs work?')
    page.locator('#send-button').click()
    expect(page.locator('.pending-message')).to_be_visible()
    page.locator('#top-library').click()
    expect(page.locator('#library-view')).to_be_visible()
    live_result=context.request.post(BASE+'/query',data={'q':'How do Kubernetes CronJobs work?'})
    assert live_result.status==200
    assert len(held_queries)==1
    held_queries[0].fulfill(status=200,content_type='application/json',body=live_result.text())
    page.unroute('**/query',hold_query)
    expect(page.locator('#send-button')).to_have_attribute('aria-label','Send question')
    page.locator('#top-library').click()
    expect(page.locator('.pending-message')).to_have_count(0)
    expect(page.locator('.message-answer .answer-content')).to_have_count(1)
    record('completed_query_visible_after_library_navigation')
    # Only this fresh browser context's test history is removed.
    page.locator('#clear-history').click()
    expect(page.locator('#clear-dialog')).to_be_visible()
    page.locator('#confirm-clear').click()
    expect(page.locator('#empty-state')).to_be_visible()
    expect(page.locator('#conversation-list .history-item')).to_have_count(0)
    record('clear_browser_history')
    # Exercise cloud UI request contracts without contacting any providers.
    cloud_queries=[]
    def cloud_health(route):
        route.fulfill(status=200,content_type='application/json',body=json.dumps({'status':'ready','mode':'cloud','document_count':11,'reference_document_count':6,'chunk_count':100,'generation':'cloud','retrieval':'Qdrant + FlashRank'}))
    def cloud_query(route):
        cloud_queries.append(route.request.post_data_json)
        route.fulfill(status=200,content_type='application/json',body=json.dumps({'answer':'Cloud request scope verified.','mode':'cloud','elapsed_ms':1,'status':'complete','sources':[],'citations':[]}))
    page.route('**/health',cloud_health)
    page.route('**/query',cloud_query)
    page.reload(wait_until='networkidle')
    expect(page.locator('#connection-label')).to_have_text('Cloud · remote index')
    expect(page.locator('#scope')).to_be_visible()
    expect(page.locator('#scope')).to_be_enabled()
    page.locator('#question').fill('How do Kubernetes CronJobs work?')
    page.locator('#send-button').click()
    expect(page.locator('.message-answer')).to_have_count(1)
    assert cloud_queries[-1]['scope']=='reference' and cloud_queries[-1]['top_k']==4
    page.locator('#scope').select_option('all')
    expect(page.locator('#scope-help')).to_contain_text('entire remote index')
    page.locator('#question').fill('How do Kubernetes Jobs work?')
    page.locator('#send-button').click()
    expect(page.locator('.message-answer')).to_have_count(2)
    assert cloud_queries[-1]['scope']=='all' and cloud_queries[-1]['top_k']==4
    page.unroute('**/health',cloud_health)
    page.unroute('**/query',cloud_query)
    record('cloud_scope_control_and_request_contract')
    assert not errors, errors
    assert not failed_requests, failed_requests
    report={'verified_at':datetime.now(timezone.utc).isoformat(),'browser_version':browser.version,'checks':checks,'page_errors':errors,'failed_requests':failed_requests,'base_url':BASE}
    (OUT/'browser_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    browser.close()
